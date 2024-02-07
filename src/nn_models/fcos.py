import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from torchvision.models.detection import fcos_resnet50_fpn, FCOS
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss, boxes as box_ops
from torchvision.models.detection import _utils as det_utils
#from ...ops import boxes as box_ops, generalized_box_iou_loss, misc as misc_nn_ops, sigmoid_focal_loss

__annotations__ = {
    "box_coder": det_utils.BoxLinearCoder,
}

def my_eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]
                        ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    if self.training:
        return (losses, detections)

    return detections


def my_postprocess_detections(
    self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]
    box_ctrness = head_outputs["bbox_ctrness"]

    num_images = len(image_shapes)

    detections: List[Dict[str, Tensor]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        box_ctrness_per_image = [bc[index] for bc in box_ctrness]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        # ---------- my changes --------------------------
        image_scores_classes = []
        image_scores_centerness = []
        # ---------------------------------------------
        image_labels = []

        for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]

            # ---------- my changes --------------------------
            class_score = torch.sigmoid(logits_per_level)
            centerness_score = torch.sigmoid(box_ctrness_per_level)
            #print("centerness max ", torch.max(centerness_score), " min ", torch.min(centerness_score))
            #print("class max ", torch.max(class_score), " min ", torch.min(class_score))
            scores_classes_per_level = class_score.flatten()
            scores_centerness_per_level = centerness_score.repeat_interleave(num_classes)

            # remove low scoring boxes
            scores_per_level = torch.sqrt(
                    class_score * centerness_score
                ).flatten()

            scores_incorrect_per_level = ((class_score - 1) * (centerness_score - 1)).flatten()

            if self.enable_unknown:
                #print("ARE U SURE YOU WANT TO USE UNKNOWN HERE ?")
                class_score = class_score.flatten()
                centerness_score = centerness_score.expand(-1, num_classes).flatten()
                keep_idxs_unknown = (centerness_score > self.centerness_unknown_threshold_supp) & (class_score < self.class_unknown_threshold_inf)
                scores_per_level[keep_idxs_unknown] = centerness_score[keep_idxs_unknown]
                keep_idxs_unknown = torch.where(keep_idxs_unknown)[0]

            if self.my_scoring_function_filter:
                #print("USE ME FUCKING FILTER FUNCTION !!! with ", self.score_thresh)
                keep_idxs = scores_incorrect_per_level < self.score_thresh
            # ---------------------------------------------

            else:
                keep_idxs = scores_per_level > self.score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            scores_centerness_per_level = scores_centerness_per_level[keep_idxs]
            scores_classes_per_level = scores_classes_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]
            scores_centerness_per_level = scores_centerness_per_level[idxs]
            scores_classes_per_level = scores_classes_per_level[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = self.box_coder.decode(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            # ------------------- my changes ------------------
            if self.enable_unknown:
                unknown_mask = torch.isin(topk_idxs, keep_idxs_unknown)
                labels_per_level[unknown_mask] = -1
            #--------------------------------------------------

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_scores_centerness.append(scores_centerness_per_level)
            image_scores_classes.append(scores_classes_per_level)
            image_labels.append(labels_per_level)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_scores_classes = torch.cat(image_scores_classes, dim=0)
        image_scores_centerness = torch.cat(image_scores_centerness, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[: self.detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
                "scores_classe": image_scores_classes[keep],
                "scores_centerness": image_scores_centerness[keep]
            }
        )

    return detections

def my_head_compute_loss(
    self,
    targets: List[Dict[str, Tensor]],
    head_outputs: Dict[str, Tensor],
    anchors: List[Tensor],
    matched_idxs: List[Tensor],
) -> Dict[str, Tensor]:

    cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
    bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
    bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

    all_gt_classes_targets = []
    all_gt_boxes_targets = []
    for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
        if len(targets_per_image["labels"]) == 0:
            gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),)) # [21486]
            gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4)) # [21486, 4]
        else:
            gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)] # [21486]
            gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)] # [21486, 4] with gt boxes associate with the corresponding gt
        gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
        all_gt_classes_targets.append(gt_classes_targets)
        all_gt_boxes_targets.append(gt_boxes_targets)

    # List[Tensor] to Tensor conversion of  `all_gt_boxes_target`, `all_gt_classes_targets` and `anchors`
    all_gt_boxes_targets, all_gt_classes_targets, anchors = (
        torch.stack(all_gt_boxes_targets),
        torch.stack(all_gt_classes_targets), # [N, 21486]
        torch.stack(anchors),
    )

    # compute foregroud
    foregroud_mask = all_gt_classes_targets >= 0 
    num_foreground = foregroud_mask.sum().detach()

    # classification loss
    gt_classes_targets = torch.zeros_like(cls_logits) # [N, HWA (21486), C]
    gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0

    # ---------------- my modification ---------------
    # Set unknown class to not matched idx with semantics that say we can have unknown
    if self.enable_semantics_classes:
        gt_classes_targets[~foregroud_mask & self.all_semantic_unknown_mask, self.all_anchors_seg_classes[~foregroud_mask & self.all_semantic_unknown_mask]] = 1

    # for logging
    self.gt_classes_targets = gt_classes_targets
    self.cls_logits = torch.sigmoid(cls_logits)

    loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction=self.class_loss_reduction) * self.class_loss_factor #TODO REFACTOR SUPRIMER TOUT CA
    # ------------------------------------------------


    #print(bbox_regression)
    # amp issue: pred_boxes need to convert float
    pred_boxes = self.box_coder.decode(bbox_regression, anchors)




    """
    all_gt_boxes_targets[:, :, 0] /= 1280
    all_gt_boxes_targets[:, :, 2] /= 1280
    all_gt_boxes_targets[:, :, 1] /= 768
    all_gt_boxes_targets[:, :, 3] /= 768
    """

    # regression loss: GIoU loss
    #print("pred : ", pred_boxes[foregroud_mask], "\n gt :", all_gt_boxes_targets[foregroud_mask], "\n diff : ", pred_boxes[foregroud_mask] - all_gt_boxes_targets[foregroud_mask],  (pred_boxes[foregroud_mask] - all_gt_boxes_targets[foregroud_mask]).mean() )
    loss_bbox_reg = generalized_box_iou_loss(
        pred_boxes[foregroud_mask],
        all_gt_boxes_targets[foregroud_mask],
        reduction="sum",
    )
    #print("loss bbox : ", loss_bbox_reg, "num foreground ", num_foreground )

    # ctrness loss
    bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)

    if len(bbox_reg_targets) == 0:
        gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
    else:
        left_right = bbox_reg_targets[:, :, [0, 2]]
        top_bottom = bbox_reg_targets[:, :, [1, 3]]
        gt_ctrness_targets = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
            * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        )

    pred_centerness = bbox_ctrness.squeeze(dim=2)

    # ---------------- my modification ---------------

    #For logging
    self.pred_centerness = torch.sigmoid(pred_centerness)
    self.gt_ctrness_targets = gt_ctrness_targets
    self.foregroud_mask = foregroud_mask
    self.last_pred_boxes = pred_boxes
    
    if self.enable_semantics_centerness:

        # Create mask of all anchors with road semantics (road, sky, vegetation, building)
        road_anchors_mask = (self.all_anchors_seg_classes == 10) | (self.all_anchors_seg_classes == 18)# | (self.all_anchors_seg_classes == 16) | (self.all_anchors_seg_classes == 12)
        self.centerness_foregroud_mask = foregroud_mask | road_anchors_mask

        # set centerness at 0 for road semantics and not foregroud
        gt_ctrness_targets[road_anchors_mask & ~foregroud_mask] = 0.0

        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[self.centerness_foregroud_mask], gt_ctrness_targets[self.centerness_foregroud_mask], reduction="sum"
        )

    else:
    # ------------------------------------------------
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[self.foregroud_mask], gt_ctrness_targets[self.foregroud_mask], reduction="sum"
        )

    return {
        "classification": loss_cls / max(1, num_foreground),
        "bbox_regression": loss_bbox_reg / max(1, num_foreground),
        "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
    }

def my_compute_loss(
    self,
    targets: List[Dict[str, Tensor]],
    head_outputs: Dict[str, Tensor],
    anchors: List[Tensor],
    num_anchors_per_level: List[int],
) -> Dict[str, Tensor]:

    # list of all boxes matched as good boxes
    matched_idxs = []
    # ------------- my change -------------
    if self.enable_semantics_classes or self.enable_semantics_centerness:
        all_semantic_unknown_mask = []
        all_anchors_seg_classes = []
    # ------------------------------------

    # For each images
    for anchors_per_image, targets_per_image in zip(anchors, targets):


        # no gt boxes then set all anchors as not matched
        if targets_per_image["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
            )
            continue

        gt_boxes = targets_per_image["boxes"]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Nx2
        anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # [HWA, 2]
        anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]

        # center sampling: anchor point must be close enough to gt center.
        # ------------- my change -------------
        if self.center_sampling:
        # ----------------------------------
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]
        
        # compute pairwise distance between N points and M boxes
        x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
        x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

        # anchor point must be inside gt
        # ------------- my change -------------
        if not self.center_sampling:
            pairwise_match = pairwise_dist.min(dim=2).values > 0
        else:
        # ------------------------------------
            pairwise_match &= pairwise_dist.min(dim=2).values > 0


        # each anchor is only responsible for certain scale range.
        lower_bound = anchor_sizes * 4
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1] :] = float("inf")
        pairwise_dist = pairwise_dist.max(dim=2).values
        pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

        # match the GT box with minimum area, if there are multiple GT matches
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
        pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
        min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
        matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

        matched_idxs.append(matched_idx)

        # ------------- my changes ------------------------------------------------------
        # for log purpose:
        self.anchor_centers = anchor_centers
        self.last_targets = targets_per_image
        self.gt_centers = gt_centers

        if self.enable_semantics_classes or self.enable_semantics_centerness:
            anchors_seg_classes = targets_per_image["seg_masks"][anchor_centers[:, 1].long(), anchor_centers[:, 0].long()] # [HWA]
            mask_unknown = (anchors_seg_classes != 255)

            all_semantic_unknown_mask.append(mask_unknown) # [HWA, C]
            all_anchors_seg_classes.append(anchors_seg_classes) 

    if self.enable_semantics_classes or self.enable_semantics_centerness:
        self.head.all_semantic_unknown_mask = torch.stack(all_semantic_unknown_mask)  # [N, HWA]
        self.head.all_anchors_seg_classes = torch.stack(all_anchors_seg_classes).long()  # [N, HWA]
    # -------------------------------------------------------------------------------

    return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

def my_forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
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
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
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
                        f"All bounding boxes should have positive height and width. Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the fcos heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        self.num_anchors_per_level = num_anchors_per_level

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)

        """
        head_outputs["bbox_regression"][:, :, 0] *= 1280
        head_outputs["bbox_regression"][:, :, 2] *= 1280
        head_outputs["bbox_regression"][:, :, 1] *= 768
        head_outputs["bbox_regression"][:, :, 3] *= 768
        """

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # compute the detections
        detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("FCOS always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)
