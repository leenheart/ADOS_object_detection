from typing import Dict, List, Optional, Tuple

import torch
import time
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation

from torchvision.models.detection import _utils as det_utils

# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList

from postprocessing_predictions import calculate_object_on_drivable_score_v2, calculate_SS_object_pourcent

def filter_proposals_iou_with_oro(
    self,
    proposals: Tensor,
    iou: Tensor,
    image_shapes: List[Tuple[int, int]],
    num_anchors_per_level: List[int],
) -> Tuple[List[Tensor], List[Tensor]]:

    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop through iou
    iou = iou.detach()
    iou = iou.reshape(num_images, -1)
    self.current_oro = self.current_oro.detach()
    self.current_oro = self.current_oro.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(iou)

    # select top_n boxes independently per level before applying nms
    # ------ my change -----------
    iou = torch.clamp(iou, 0, 1)
    self.current_oro = torch.clamp(self.current_oro, 0, 1)
    # ----------------------
    top_n_idx = self._get_top_n_idx(iou, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    iou = iou[batch_idx, top_n_idx]
    self.current_oro = self.current_oro[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]
    # ------ my change -----------
    current_anchors = self.current_anchors[batch_idx, top_n_idx]

    final_anchors = []
    # ----------------------
    final_boxes = []
    final_scores = []
    final_oro = []
    for boxes, scores, lvl, img_shape, anchors, oro in zip(proposals, iou, levels, image_shapes, current_anchors, self.current_oro):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        anchors = box_ops.clip_boxes_to_image(anchors, img_shape)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, self.min_size)
        boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]
        oro = oro[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= self.score_thresh)[0]
        boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]
        oro = oro[keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: self.post_nms_top_n()]
        boxes, scores, anchors = boxes[keep], scores[keep], anchors[keep]
        oro = oro[keep]

        final_anchors.append(anchors)
        final_boxes.append(boxes)
        final_scores.append(scores)
        final_oro.append(oro)

    self.current_filtered_anchors = final_anchors
    self.current_oro = final_oro

    return final_boxes, final_scores

def roi_head_postprocess_detections_without_removing_background_classes_with_oro(
    self,
    class_logits,  # type: Tensor
    box_regression,  # type: Tensor
    proposals,  # type: List[Tensor]
    image_shapes,  # type: List[Tuple[int, int]]
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
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
    all_proposals_scores = []
    all_oro_proposals_scores = []
    for boxes, scores, image_shape, proposal_scores, oro_proposal_scores in zip(pred_boxes_list, pred_scores_list, image_shapes, self.proposals_scores, self.oro_proposals_scores):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        proposal_scores = proposal_scores.view(-1, 1).expand_as(scores)
        oro_proposal_scores = oro_proposal_scores.view(-1, 1).expand_as(scores)

        # Multiply background scores by the proposal score to make even comparaison on score for known classes and background classes
        scores[:, 0] = proposal_scores[:, 0]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        proposal_scores = proposal_scores.reshape(-1)
        oro_proposal_scores = oro_proposal_scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        proposal_scores = proposal_scores[inds]
        oro_proposal_scores = oro_proposal_scores[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        proposal_scores = proposal_scores[keep]
        oro_proposal_scores = oro_proposal_scores[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        proposal_scores = proposal_scores[keep]
        oro_proposal_scores = oro_proposal_scores[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_proposals_scores.append(proposal_scores)
        all_oro_proposals_scores.append(oro_proposal_scores) 

    self.all_oro_scores = all_oro_proposals_scores
    self.all_iou_scores = all_proposals_scores

    return all_boxes, all_scores, all_labels


def my_rcnn_forward_with_oro(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    if self.training:
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, proposal_losses = self.rpn(images, features, targets)
    # my changes --------------------------
    self.roi_heads.proposals_scores = self.rpn.current_scores
    self.roi_heads.oro_proposals_scores = self.rpn.current_oro
    # ------------------------------------
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)

# got
class RPNHead_IoU_ORO(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.iou_pred = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.oro_pred = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        iou_reg = []
        oro_reg = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            iou_reg.append(self.iou_pred(t))
            oro_reg.append(self.oro_pred(t))
            bbox_reg.append(self.bbox_pred(t))
        return iou_reg, bbox_reg, oro_reg

def my_rcnn_forward(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    if self.training:
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, proposal_losses = self.rpn(images, features, targets)
    # my changes --------------------------
    self.roi_heads.proposals_scores = self.rpn.current_scores
    # ------------------------------------
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)

def roi_head_filter(roi_head, boxes, scores, labels):

    # batch everything, by making every class prediction be a separate instance
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    inds = torch.where(scores > roi_head.score_thresh)[0]
    boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

    # remove empty boxes
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # non-maximum suppression, independently done per class
    keep = box_ops.batched_nms(boxes, scores, labels, roi_head.nms_thresh)

    # keep only topk scoring predictions
    keep = keep[: roi_head.detections_per_img]

    return boxes[keep], scores[keep], labels[keep]

def roi_head_postprocess_detections_without_removing_background_classes(
    self,
    class_logits,  # type: Tensor
    box_regression,  # type: Tensor
    proposals,  # type: List[Tensor]
    image_shapes,  # type: List[Tuple[int, int]]
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
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
    for boxes, scores, image_shape, proposal_scores in zip(pred_boxes_list, pred_scores_list, image_shapes, self.proposals_scores):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        proposal_scores = proposal_scores.view(-1, 1).expand_as(scores)

        """
        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        """

        known_boxes, known_scores, known_labels = roi_head_filter(self, boxes[:, 1:], scores[:, 1:], labels[:, 1:])
        unknown_boxes, unknown_scores, unknown_labels = roi_head_filter(self, boxes[:, 0].unsqueeze(1), proposal_scores[:, 0].unsqueeze(1), labels[:, 0].unsqueeze(1))
        boxes = torch.cat((known_boxes, unknown_boxes), dim=0)
        scores = torch.cat((known_scores, unknown_scores), dim=0)
        labels = torch.cat((known_labels, unknown_labels), dim=0)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels

#"""
class RPNHead_IoU(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    _version = 2

    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.iou_pred = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for type in ["weight", "bias"]:
                old_key = f"{prefix}conv.{type}"
                new_key = f"{prefix}conv.0.0.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        iou_reg = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            iou_reg.append(self.iou_pred(t))
            bbox_reg.append(self.bbox_pred(t))
        return iou_reg, bbox_reg


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers_oro(box_cls: List[Tensor], box_regression: List[Tensor], oro: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    oro_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level, oro_per_level in zip(box_cls, box_regression, oro):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

        oro_per_level = permute_and_flatten(oro_per_level, N, A, C, H, W)
        oro_flattened.append(oro_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_oro = torch.cat(oro_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression, box_oro

def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def compute_loss_iou_for_objecteness(
        self, ious: Tensor, pred_bbox_deltas: Tensor, ious_targets: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            iou (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            iou_loss (Tensor)
            box_loss (Tensor)
        """
        
        ious = ious.flatten()

        sampled_inds_iou = []
        sampled_inds_bbox = []
        self.current_sampled_anchors = []
        self.current_sampled_iou_targets = []
        self.current_sampled_iou_predictions = []
        ious_index = 0
        for anchors_per_image, iou_targets_per_image in zip(self.current_anchors, ious_targets):


            positive_inds_iou = torch.where(iou_targets_per_image >= 0.3)[0]
            positive_inds_bbox = torch.where(iou_targets_per_image >= 0.7)[0]

            perm_iou = torch.randperm(positive_inds_iou.numel(), device=positive_inds_iou.device)[:self.batch_size_per_image]
            perm_bbox = torch.randperm(positive_inds_bbox.numel(), device=positive_inds_bbox.device)[:self.batch_size_per_image]

            pos_idx_per_image_iou = positive_inds_iou[perm_iou]
            pos_idx_per_image_mask_iou = torch.zeros_like(iou_targets_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask_iou[pos_idx_per_image_iou] = 1

            pos_idx_per_image_bbox = positive_inds_bbox[perm_bbox]
            pos_idx_per_image_mask_bbox = torch.zeros_like(iou_targets_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask_bbox[pos_idx_per_image_bbox] = 1

            sampled_inds_iou.append(pos_idx_per_image_mask_iou)
            sampled_inds_bbox.append(pos_idx_per_image_mask_bbox)

            self.current_sampled_anchors.append(anchors_per_image[pos_idx_per_image_mask_iou])
            self.current_sampled_iou_targets.append(iou_targets_per_image[pos_idx_per_image_mask_iou])
            self.current_sampled_iou_predictions.append(ious[ious_index:ious_index + len(anchors_per_image)][pos_idx_per_image_mask_iou])
            ious_index += len(anchors_per_image)

        sampled_inds_iou = torch.where(torch.cat(sampled_inds_iou, dim=0))[0]
        sampled_inds_bbox = torch.where(torch.cat(sampled_inds_bbox, dim=0))[0]

        ious_targets = torch.cat(ious_targets, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_inds_bbox],
            regression_targets[sampled_inds_bbox],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_bbox.numel())

        ious_loss = F.smooth_l1_loss(
            ious[sampled_inds_iou],
            ious_targets[sampled_inds_iou],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_iou.numel())

        return ious_loss, box_loss

def assign_targets_to_anchors_iou(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        ious = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                ious_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)

                # match_quality_matrix is M (gt) x N (predicted)
                # Max over gt elements (dim 0) to find best gt candidate for each prediction
                matched_vals, matches = match_quality_matrix.max(dim=0)


                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matches.clamp(min=0)]
                ious_per_image = matched_vals


            ious.append(ious_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return ious, matched_gt_boxes


def filter_proposals_iou(
    self,
    proposals: Tensor,
    iou: Tensor,
    image_shapes: List[Tuple[int, int]],
    num_anchors_per_level: List[int],
) -> Tuple[List[Tensor], List[Tensor]]:

    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop through iou
    iou = iou.detach()
    iou = iou.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(iou)

    # select top_n boxes independently per level before applying nms
    # ------ my change -----------
    iou = torch.clamp(iou, 0, 1)
    # ----------------------
    top_n_idx = self._get_top_n_idx(iou, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    iou = iou[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]
    # ------ my change -----------
    current_anchors = self.current_anchors[batch_idx, top_n_idx]
    # ----------------------


    # ------ my change -----------
    final_anchors = []
    # ----------------------
    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape, anchors in zip(proposals, iou, levels, image_shapes, current_anchors):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        anchors = box_ops.clip_boxes_to_image(anchors, img_shape)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, self.min_size)
        boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= self.score_thresh)[0]
        boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: self.post_nms_top_n()]
        boxes, scores, anchors = boxes[keep], scores[keep], anchors[keep]

        final_anchors.append(anchors)
        final_boxes.append(boxes)
        final_scores.append(scores)

    self.current_filtered_anchors = final_anchors

    return final_boxes, final_scores

def filter_proposals_that_save_infos_to_be_log(
    self,
    proposals: Tensor,
    objectness: Tensor,
    image_shapes: List[Tuple[int, int]],
    num_anchors_per_level: List[int],
) -> Tuple[List[Tensor], List[Tensor]]:

    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop through objectness
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    # select top_n boxes independently per level before applying nms
    top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]
    # ------ my change -----------
    current_anchors = self.current_anchors[batch_idx, top_n_idx]
    # ----------------------

    objectness_prob = torch.sigmoid(objectness)

    # ------ my change -----------
    final_anchors = []
    # ----------------------
    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape, anchors in zip(proposals, objectness_prob, levels, image_shapes, current_anchors):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
        anchors = box_ops.clip_boxes_to_image(anchors, img_shape)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, self.min_size)
        boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= self.score_thresh)[0]
        boxes, scores, lvl, anchors = boxes[keep], scores[keep], lvl[keep], anchors[keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: self.post_nms_top_n()]
        boxes, scores, anchors = boxes[keep], scores[keep], anchors[keep]

        final_anchors.append(anchors)
        final_boxes.append(boxes)
        final_scores.append(scores)

    self.current_filtered_anchors = final_anchors

    return final_boxes, final_scores

def rpn_forward_that_save_infos_to_be_log(
    self,
    images: ImageList,
    features: Dict[str, Tensor],
    targets: Optional[List[Dict[str, Tensor]]] = None,
) -> Tuple[List[Tensor], Dict[str, Tensor]]:

    """
    Args:
        images (ImageList): images for which we want to compute the predictions
        features (Dict[str, Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
        targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
            If provided, each element in the dict should contain a field `boxes`,
            with the locations of the ground-truth boxes.

    Returns:
        boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            image.
        losses (Dict[str, Tensor]): the losses for the model during training. During
            testing, it is an empty dict.
    """
    # RPN uses all feature maps that are available
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    # ------ my change -----------
    self.current_anchors = torch.stack(anchors)
    # ----------------------

    boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    # ------ my change -----------
    self.current_scores = scores
    self.current_proposals = boxes
    self.current_anchors = anchors
    # ----------------------

    losses = {}
    if self.training:
        if targets is None:
            raise ValueError("targets should not be None")
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        # ------ my change -----------
        self.current_labels = labels 
        # ------ -----------
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    return boxes, losses

def compute_loss_oro_iou_for_objecteness_with_negative_sample(
        self, ious: Tensor, oros: Tensor, pred_bbox_deltas: Tensor, ious_targets: List[Tensor], oros_ss_targets: List[Tensor], anchors: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        ious = ious.flatten()
        oros = oros.flatten()

        sampled_inds_iou = []
        sampled_inds_bbox = []
        self.current_sampled_anchors = []
        self.current_sampled_iou_targets = []
        self.current_sampled_iou_predictions = []
        index = 0

        oros_targets = []
        oros_predictions = []

        for anchors_per_image, iou_targets_per_image, ss_target in zip(self.current_anchors, ious_targets, oros_ss_targets):

            # Select positive depending on their iou
            if self.add_best_iou_sample:
                positive_inds_iou = torch.where((iou_targets_per_image >= 0.3) & (iou_targets_per_image < 0.7))[0]
            else:
                positive_inds_iou = torch.where(iou_targets_per_image >= 0.3)[0]
            negative_inds_iou = torch.where(iou_targets_per_image < 0.3)[0]
            positive_inds_bbox = torch.where(iou_targets_per_image >= 0.7)[0]


            # select 1000 boxes to not take too long to calculate the SS scores
            perm_iou_negative = torch.randperm(negative_inds_iou.numel(), device=negative_inds_iou.device)[:self.batch_size_per_image * 2]

            # calculate SS scores on than selected anchors
            percent_of_object_negative_sample = calculate_SS_object_pourcent(anchors_per_image[negative_inds_iou[perm_iou_negative]], ss_target, display=False, training=True)

            # select negative that have 0 percent of object in it :
            perm_iou_negative = perm_iou_negative[percent_of_object_negative_sample == 0][:int(self.batch_size_per_image * self.percent_of_negative_sample)]

            # make iou target of negative at 0
            iou_targets_per_image[negative_inds_iou[perm_iou_negative]] = 0


            # take a random selection on those
            perm_bbox = torch.randperm(positive_inds_bbox.numel(), device=positive_inds_bbox.device)[:self.batch_size_per_image]

            number_of_positive_sample_iou = int(self.batch_size_per_image * (1 - self.percent_of_negative_sample))
            if self.add_best_iou_sample:
                perm_best_iou = torch.randperm(positive_inds_bbox.numel(), device=positive_inds_bbox.device)[:number_of_positive_sample_iou]
                perm_iou_positive = torch.randperm(positive_inds_iou.numel(), device=positive_inds_iou.device)[:number_of_positive_sample_iou - min(number_of_positive_sample_iou, perm_best_iou.numel())]
                sampled_idx_per_image_iou = torch.cat([positive_inds_iou[perm_iou_positive], negative_inds_iou[perm_iou_negative], positive_inds_bbox[perm_best_iou]], dim=0)
            else:
                perm_iou_positive = torch.randperm(positive_inds_iou.numel(), device=positive_inds_iou.device)[:number_of_positive_sample_iou]
                sampled_idx_per_image_iou = torch.cat([positive_inds_iou[perm_iou_positive], negative_inds_iou[perm_iou_negative]], dim=0)
            sampled_idx_per_image_mask_iou = torch.zeros_like(iou_targets_per_image, dtype=torch.uint8)
            sampled_idx_per_image_mask_iou[sampled_idx_per_image_iou] = 1

            pos_idx_per_image_bbox = positive_inds_bbox[perm_bbox]
            pos_idx_per_image_mask_bbox = torch.zeros_like(iou_targets_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask_bbox[pos_idx_per_image_bbox] = 1

            sampled_inds_iou.append(sampled_idx_per_image_mask_iou)
            sampled_inds_bbox.append(pos_idx_per_image_mask_bbox)

            self.current_sampled_anchors.append(anchors_per_image[sampled_idx_per_image_mask_iou])
            self.current_sampled_iou_targets.append(iou_targets_per_image[sampled_idx_per_image_mask_iou])
            self.current_sampled_iou_predictions.append(ious[index:index + len(anchors_per_image)][sampled_idx_per_image_mask_iou])
            oros_predictions.append(oros[index:index + len(anchors_per_image)][sampled_idx_per_image_mask_iou])
            index += len(anchors_per_image)

            oros_targets.append(calculate_object_on_drivable_score_v2(anchors_per_image[sampled_idx_per_image_mask_iou], ss_target, display=False, training=True))



        self.current_oro_targets = oros_targets
        self.current_oro_predictions = oros_predictions
        self.current_oro_sampled_anchors = self.current_sampled_anchors

        sampled_inds_iou = torch.where(torch.cat(sampled_inds_iou, dim=0))[0]
        sampled_inds_bbox = torch.where(torch.cat(sampled_inds_bbox, dim=0))[0]

        ious_targets = torch.cat(ious_targets, dim=0)
        oros_targets = torch.cat(oros_targets, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_inds_bbox],
            regression_targets[sampled_inds_bbox],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_bbox.numel())

        ious_loss = F.smooth_l1_loss(
            ious[sampled_inds_iou],
            ious_targets[sampled_inds_iou],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_iou.numel())

        oro_loss = F.smooth_l1_loss(
            oros[sampled_inds_iou],
            oros_targets,
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_iou.numel()) 

        return ious_loss, oro_loss, box_loss

def compute_loss_oro_iou_for_objecteness(
        self, ious: Tensor, oros: Tensor, pred_bbox_deltas: Tensor, ious_targets: List[Tensor], oros_ss_targets: List[Tensor], anchors: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        ious = ious.flatten()
        oros = oros.flatten()

        sampled_inds_iou = []
        sampled_inds_bbox = []
        self.current_sampled_anchors = []
        self.current_sampled_iou_targets = []
        self.current_sampled_iou_predictions = []
        index = 0

        oros_targets = []
        oros_predictions = []

        for anchors_per_image, iou_targets_per_image, ss_target in zip(self.current_anchors, ious_targets, oros_ss_targets):

            positive_inds_iou = torch.where(iou_targets_per_image >= 0.3)[0]
            positive_inds_bbox = torch.where(iou_targets_per_image >= 0.7)[0]

            perm_iou = torch.randperm(positive_inds_iou.numel(), device=positive_inds_iou.device)[:self.batch_size_per_image]
            perm_bbox = torch.randperm(positive_inds_bbox.numel(), device=positive_inds_bbox.device)[:self.batch_size_per_image]

            pos_idx_per_image_iou = positive_inds_iou[perm_iou]
            pos_idx_per_image_mask_iou = torch.zeros_like(iou_targets_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask_iou[pos_idx_per_image_iou] = 1

            pos_idx_per_image_bbox = positive_inds_bbox[perm_bbox]
            pos_idx_per_image_mask_bbox = torch.zeros_like(iou_targets_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask_bbox[pos_idx_per_image_bbox] = 1

            sampled_inds_iou.append(pos_idx_per_image_mask_iou)
            sampled_inds_bbox.append(pos_idx_per_image_mask_bbox)

            self.current_sampled_anchors.append(anchors_per_image[pos_idx_per_image_mask_iou])
            self.current_sampled_iou_targets.append(iou_targets_per_image[pos_idx_per_image_mask_iou])
            self.current_sampled_iou_predictions.append(ious[index:index + len(anchors_per_image)][pos_idx_per_image_mask_iou])
            oros_predictions.append(oros[index:index + len(anchors_per_image)][pos_idx_per_image_mask_iou])
            index += len(anchors_per_image)

            oros_targets.append(calculate_SS_object_pourcent(anchors_per_image[pos_idx_per_image_mask_iou], ss_target, display=False, training=True))



        self.current_oro_targets = oros_targets
        self.current_oro_predictions = oros_predictions
        self.current_oro_sampled_anchors = self.current_sampled_anchors

        sampled_inds_iou = torch.where(torch.cat(sampled_inds_iou, dim=0))[0]
        sampled_inds_bbox = torch.where(torch.cat(sampled_inds_bbox, dim=0))[0]

        ious_targets = torch.cat(ious_targets, dim=0)
        oros_targets = torch.cat(oros_targets, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_inds_bbox],
            regression_targets[sampled_inds_bbox],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_bbox.numel())

        ious_loss = F.smooth_l1_loss(
            ious[sampled_inds_iou],
            ious_targets[sampled_inds_iou],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_iou.numel())

        oro_loss = F.smooth_l1_loss(
            oros[sampled_inds_iou],
            oros_targets,
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds_iou.numel()) 

        return ious_loss, oro_loss, box_loss



def rpn_forward_oro_that_save_infos_to_be_log(
    self,
    images: ImageList,
    features: Dict[str, Tensor],
    targets: Optional[List[Dict[str, Tensor]]] = None,
) -> Tuple[List[Tensor], Dict[str, Tensor]]:

    """
    Args:
        images (ImageList): images for which we want to compute the predictions
        features (Dict[str, Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
        targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
            If provided, each element in the dict should contain a field `boxes`,
            with the locations of the ground-truth boxes.

    Returns:
        boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            image.
        losses (Dict[str, Tensor]): the losses for the model during training. During
            testing, it is an empty dict.
    """
    # RPN uses all feature maps that are available
    features = list(features.values())
    # ------ my change -----------
    iou, pred_bbox_deltas, oro = self.head(features)

    # ----------------------
    anchors = self.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in iou]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    iou, pred_bbox_deltas, oro = concat_box_prediction_layers_oro(iou, pred_bbox_deltas, oro)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)

    # ------ my change -----------
    self.current_anchors = torch.stack(anchors)
    self.current_oro = oro
    # ----------------------

    boxes, scores = self.filter_proposals(proposals, iou, images.image_sizes, num_anchors_per_level)

    # ------ my change -----------
    self.current_scores = scores
    self.current_iou = iou
    self.current_proposals = boxes
    self.current_anchors = anchors
    # ----------------------

    losses = {}
    if self.training:

        if targets is None:
            raise ValueError("targets should not be None")
        iou_targets, ss_targets, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        # ------ my change -----------
        self.current_iou_targets = iou_targets

        loss_iou, loss_oro, loss_rpn_box_reg = self.compute_loss(
            iou, oro, pred_bbox_deltas, iou_targets, ss_targets, anchors, regression_targets
        )
        losses = {
            "loss_iou": loss_iou,
            "loss_oro": loss_oro,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        # ----------------------
    return boxes, losses

def assign_targets_to_anchors_iou_oro(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

        ious_targets = []
        oros_targets = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                ious_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
                oros_per_image = targets_per_image["semantic_segmentation_OBD"]
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)

                # match_quality_matrix is M (gt) x N (predicted)
                # Max over gt elements (dim 0) to find best gt candidate for each prediction
                matched_vals, matches = match_quality_matrix.max(dim=0)

                threshold_minimum_iou = 0.3
                negative_mask = matched_vals < threshold_minimum_iou

                matched_vals[negative_mask] = -1

                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matches.clamp(min=0)]
                ious_per_image = matched_vals
                oros_per_image = targets_per_image["semantic_segmentation_OBD"]


            ious_targets.append(ious_per_image)
            oros_targets.append(oros_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return ious_targets, oros_targets, matched_gt_boxes
