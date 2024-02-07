from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from torchvision.ops import box_area
from torchvision.ops._utils import _upcast
from torchvision.models.detection import _utils as det_utils

# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList

from postprocessing_predictions import calculate_object_on_drivable_score_v2, calculate_SS_object_pourcent


def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
                    

def roi_head_filter(roi_head, boxes, scores, labels, proposal_scores=None, oro_proposal_scores=None):

    # batch everything, by making every class prediction be a separate instance
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if roi_head.with_oro:
        proposal_scores = proposal_scores.reshape(-1)
        oro_proposal_scores = oro_proposal_scores.reshape(-1)

    # remove low scoring boxes
    inds = torch.where(scores > roi_head.score_thresh)[0]
    boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
    if roi_head.with_oro:
        proposal_scores = proposal_scores[inds]
        oro_proposal_scores = oro_proposal_scores[inds]

    # remove empty boxes
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if roi_head.with_oro:
        proposal_scores = proposal_scores[keep]
        oro_proposal_scores = oro_proposal_scores[keep]


    # non-maximum suppression, independently done per class
    keep = box_ops.batched_nms(boxes, scores, labels, roi_head.nms_thresh)

    # keep only topk scoring predictions
    keep = keep[: roi_head.detections_per_img]

    if roi_head.with_oro:
        return boxes[keep], scores[keep], labels[keep], proposal_scores[keep], oro_proposal_scores[keep]
    return boxes[keep], scores[keep], labels[keep]

def my_roi_head_postprocess_detections(
    self,
    class_logits,  # type: Tensor
    box_regression,  # type: Tensor
    proposals,  # type: List[Tensor]
    image_shapes,  # type: List[Tuple[int, int]]
):
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    if self.with_oro:
        all_proposals_scores = []
        all_oro_proposals_scores = []

    for i, (boxes, scores, image_shape, proposal_scores) in enumerate(zip(pred_boxes_list, pred_scores_list, image_shapes, self.proposals_scores)):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        proposal_scores = proposal_scores.view(-1, 1).expand_as(scores)

        # oro case
        if self.with_oro:

            # Getting known objects
            oro_proposal_scores = self.oro_proposals_scores[i].view(-1, 1).expand_as(scores)

            known_boxes, known_scores, known_labels, known_proposal_scores, known_oro_proposal_scores = roi_head_filter(self, boxes[:, 1:], scores[:, 1:], labels[:, 1:], proposal_scores=proposal_scores[:, 1:], oro_proposal_scores=oro_proposal_scores[: , 1:])

            # Getting mix unknown and background objects
            if self.keep_background:


                # remove all those that does not have big background scores
                unknown_keeps_score = torch.where((scores[:, 0] > self.unknown_roi_head_background_classif_score_threshold) & (oro_proposal_scores[:, 0] > self.unknown_roi_head_oro_score_threshold ) & (proposal_scores[:, 0] > self.unknown_roi_head_iou_score_threshold))[0]


                unknown_keeps_outside_known = scores[:, 0][unknown_keeps_score] >= 0

                scores[:, 0] = proposal_scores[:, 0]
                self.detections_per_img = self.unknown_detections_per_img
                unknown_boxes, unknown_scores, unknown_labels, unknown_proposal_scores, unknown_oro_proposal_scores = roi_head_filter(self, boxes[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1), proposal_scores[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1), labels[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1), proposal_scores=proposal_scores[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1), oro_proposal_scores=oro_proposal_scores[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1))

                self.detections_per_img = self.known_detections_per_img

                # Keep known or mix objects scores depending on settings
                proposal_scores = torch.cat((known_proposal_scores, unknown_proposal_scores), dim=0)
                oro_proposal_scores = torch.cat((known_oro_proposal_scores, unknown_oro_proposal_scores), dim=0)
            else:
                proposal_scores = known_proposal_scores
                oro_proposal_scores = known_oro_proposal_scores

        # basic case
        else:
            # Getting known objects
            known_boxes, known_scores, known_labels = roi_head_filter(self, boxes[:, 1:], scores[:, 1:], labels[:, 1:])

            # Getting mix unknown and background objects
            if self.keep_background:

                unknown_keeps_score = torch.where((scores[:, 0] > self.unknown_roi_head_background_classif_score_threshold))# & (proposal_scores[:, 0] > self.unknown_roi_head_iou_score_threshold))[0]

                unknown_keeps_outside_known = scores[:, 0][unknown_keeps_score] >= 0

                scores[:, 0] = proposal_scores[:, 0]
                self.detections_per_img = self.unknown_detections_per_img

                unknown_boxes, unknown_scores, unknown_labels = roi_head_filter(self, boxes[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1), proposal_scores[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1), labels[:, 0][unknown_keeps_score][unknown_keeps_outside_known].unsqueeze(1))
                self.detections_per_img = self.known_detections_per_img


        # Keep known or mix objects depending on settings
        if self.keep_background:
            boxes = torch.cat((known_boxes, unknown_boxes), dim=0)
            scores = torch.cat((known_scores, unknown_scores), dim=0)
            labels = torch.cat((known_labels, unknown_labels), dim=0)
        else:
            boxes = known_boxes
            labels = known_labels
            scores = known_scores

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        if self.with_oro:
            all_proposals_scores.append(proposal_scores)
            all_oro_proposals_scores.append(oro_proposal_scores) 

    if self.with_oro:
        self.all_oro_scores = all_oro_proposals_scores
        self.all_iou_scores = all_proposals_scores

    return all_boxes, all_scores, all_labels
