from typing import Tuple

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import wandb
import os
import time
import numpy as np
import json
import cv2 as cv
import copy

from datetime import datetime

# externals Models
from torchvision.models.detection.fcos import fcos_resnet50_fpn, FCOS, FCOSHead
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, _default_anchorgen
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN 
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torch import Tensor
from ultralytics import YOLO
from torchvision.ops import box_area

from datamodule import DataModule 
from plot_data import fig2rgb_array
from metrics import MetricModule
from log import get_wandb_image_with_labels, get_wandb_image_with_labels_background_unknown_known, get_wandb_image_with_labels_target_known_unknown_background, get_wandb_image_with_proposal
from postprocessing_predictions import postprocess_predictions
from postprocessing_predictions import set_tags, seperate_predictions_with_threshold_score, seperate_predictions_into_known_and_unknown, get_only_known_targets, get_only_unknown_targets, get_only_background_targets

from nn_models.fcos import my_forward, my_eager_outputs, my_compute_loss, my_head_compute_loss, my_postprocess_detections
from nn_models.my_rpn import compute_loss_iou_for_objecteness, assign_targets_to_anchors_iou, filter_proposals_iou, rpn_forward_that_save_infos_to_be_log, filter_proposals_that_save_infos_to_be_log, my_rcnn_forward, assign_targets_to_anchors_iou_oro, compute_loss_oro_iou_for_objecteness, rpn_forward_oro_that_save_infos_to_be_log, my_rcnn_forward_with_oro, filter_proposals_iou_with_oro, RPNHead_IoU_ORO, RPNHead_IoU
from nn_models.my_rpn import RPNHead_IoU_ORO, compute_loss_oro_iou_for_objecteness_with_negative_sample 
from nn_models.my_roi_heads import my_roi_head_postprocess_detections, _box_inter_union


# CODE FROM https://github.com/hustvl/YOLOP/blob/main/lib/core/general.py#L98 use for yolop
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# CODE FROM https://github.com/hustvl/YOLOP/blob/main/lib/core/general.py#L98 use for yolop
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def convert_yolop_prediction_format(predictions):

    # from list of yolop results to list of dict with "boxes", "scores", "labels", "score_classe", "score_centerness"
    # boxes is a tensor of (n, 4)
    # labels and scores is a tensor of n
    # doc for yolop results is : https://github.com/hustvl/YOLOP/blob/main/lib/core/general.py
    # the prediction input is a list of n * 6 with 6 being [xyxy, confidence score, classe]


    converted_predictions = []
    for prediction in predictions:
        converted_prediction = {}
        converted_prediction["boxes"] = prediction[:, :4]
        converted_prediction["labels"] = prediction[:, 5].int()
        converted_prediction["scores"] = prediction[:, 4]

        converted_predictions.append(converted_prediction)

    return converted_predictions
                   

def convert_yolo_prediction_format(predictions):

    # from list of yolo results to list of dict with "boxes", "scores", "labels", "score_classe", "score_centerness"
    # boxes is a tensor of (n, 4)
    # labels and scores is a tensor of n
    # doc for yolo results is : https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes and https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor.to 


    converted_predictions = []
    for prediction in predictions:
        converted_prediction = {}
        converted_prediction["boxes"] = prediction.boxes.xyxy
        converted_prediction["labels"] = prediction.boxes.cls.int()
        converted_prediction["scores"] = prediction.boxes.conf

        converted_predictions.append(converted_prediction)

    return converted_predictions
                    
"""

Create a pytorch lightning module with fcos model from pytorch

"""
class Model(pl.LightningModule):

    def init_modified_fcos(self, cfg_model):

        # Create pytorch implementation of fcos
        if cfg_model.pretrained:
            self.model = fcos_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        else:
            self.model = fcos_resnet50_fpn(num_classes=self.num_classes, _skip_resize=True)

        # Change pytorch implementation toward mine
        FCOS.forward = my_forward
        FCOS.eager_outputs = my_eager_outputs # to get model output at the same time than the losses at training time
        FCOS.compute_loss = my_compute_loss
        FCOS.postprocess_detections = my_postprocess_detections
        FCOSHead.compute_loss = my_head_compute_loss

        # Add parameters to the fcos model to adapte my changes
        self.model.enable_unknown = cfg_model.enable_unknown

        self.model.class_unknown_threshold_inf = self.threshold_score_centerness_unknown
        self.model.centerness_unknown_threshold_supp = self.threshold_score_remove

        self.model.center_sampling = self.center_sampling
        self.model.enable_semantics_classes = cfg_model.enable_semantics_classes
        self.model.enable_semantics_centerness = cfg_model.enable_semantics_centerness
        self.model.head.enable_semantics_classes = cfg_model.enable_semantics_classes
        self.model.head.enable_semantics_centerness = cfg_model.enable_semantics_centerness
        self.model.head.nb_classes = self.num_classes
        self.model.head.class_loss_reduction = cfg_model.class_loss_reduction
        self.model.head.class_loss_factor = cfg_model.class_loss_factor
        self.model.my_scoring_function_filter = cfg_model.my_scoring_function_filter
        self.model.score_thresh = cfg_model.object_score_threshold

    def __init__(self, cfg_model, cfg_metrics, classes_known, classes_names_pred=None, classes_names_gt=None, classes_names_merged=None, batch_size=1, show_image=False, use_custom_scores=False, semantic_segmentation_class_id_label=None ):

        super().__init__()

        self.use_custom_scores = use_custom_scores
        self.scores_cfg = cfg_metrics.scores
        self.considering_known_classes = cfg_metrics.considering_known_classes 
        self.classes_known = classes_known 

        self.test_metrics = MetricModule(self.scores_cfg, cfg_metrics.testing, len(classes_names_pred), len(classes_names_gt), classes_known)
        self.val_metrics = MetricModule(self.scores_cfg, cfg_metrics.validation, len(classes_names_pred), len(classes_names_gt), classes_known)

        self.num_classes = len(classes_names_pred)

        self.save_predictions_path = cfg_model.save_predictions_path
        self.save_predictions = cfg_model.save_predictions

        self.threshold_score_centerness_unknown = cfg_model.threshold_score_centerness_unknown
        self.threshold_score_remove = cfg_model.threshold_score_remove 
        self.threshold_score_good = cfg_model.threshold_score_good 

        self.best_map = 0

        self.load_path = cfg_model.load_path
        if cfg_model.load:
            self.load(cfg_model.to_load)

        # FCOS
        if cfg_model.name == "fcos":
            if cfg_model.pretrained and not cfg_model.load:
                self.model = torchvision.models.detection.fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
                print("Using pretrained model on coco")
            else:
                raise NotImplementedError("Fcos modifed is implemented but not up to date with the rest of the code")
                print("Using My fcos modified model")
                self.center_sampling = cfg_model.center_sampling
                self.enable_semantics_centerness = cfg_model.enable_semantics_centerness
                self.log_heatmap = cfg_model.log_heatmap
                self.log_fcos_inside = cfg_model.log_fcos_inside 
                self.init_modified_fcos(cfg_model)

        # Retina Net
        elif cfg_model.name == "retina_net":
            if cfg_model.pretrained and not cfg_model.load:
                self.model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
            else:
                raise NotImplementedError("This function is not yet implemented")
                self.model = retinanet_resnet50_fpn_v2(num_classes=self.num_classes)

        # Faster-RCNN
        elif cfg_model.name == "faster_rcnn":
            if cfg_model.pretrained and not cfg_model.load:
                self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
            elif not cfg_model.load:
                raise NotImplementedError("This function is not yet implemented")


            self.model.roi_heads.detections_per_img = cfg_model.known_detections_per_img
            self.model.roi_heads.known_detections_per_img = cfg_model.known_detections_per_img
            self.model.roi_heads.unknown_detections_per_img = cfg_model.unknown_detections_per_img
            self.model.roi_heads.keep_background = cfg_model.keep_background
            self.model.roi_heads.with_oro = cfg_model.ai_oro
            self.model.roi_heads.unknown_intersection_with_known_threshold = cfg_model.unknown_intersection_with_known_threshold

            RoIHeads.postprocess_detections = my_roi_head_postprocess_detections
            GeneralizedRCNN.forward = my_rcnn_forward

            if cfg_model.keep_background:
                print("\nKeep the background class in post processing of roi head\n")


            if cfg_model.reset_rpn_weight:

                print("\n Reset rpn weight !\n")
                model_random = fasterrcnn_resnet50_fpn_v2()
                self.model.rpn = model_random.rpn

            if cfg_model.freeze_backbone:
                # Freeze backbone and roi heads:
                print("\nFreeze Backbone \n")
                for p in self.model.backbone.parameters():
                    p.requires_grad = False

            if cfg_model.freeze_heads:
                print("\nFreeze ROI Heads\n")
                for p in self.model.roi_heads.parameters():
                    p.requires_grad = False

            if cfg_model.freeze_rpn_heads:

                print("rpn head tyep :", type(self.model.rpn.head))
                if isinstance(self.model.rpn.head, RPNHead_IoU) and cfg_model.ai_oro:
                    print("Changing RPN HEAD IoU by ORO with keeping the weights")
                    out_channels = self.model.backbone.out_channels
                    rpn_anchor_generator = _default_anchorgen()
                    rpn_head = RPNHead_IoU_ORO(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

                    rpn_head.conv = self.model.rpn.head.conv
                    rpn_head.iou_pred = self.model.rpn.head.iou_pred
                    rpn_head.bbox_pred = self.model.rpn.head.bbox_pred
                    self.model.rpn.head = rpn_head


                print("\nFreeze RPN Heads\n")
                for p in self.model.rpn.head.conv.parameters():
                    p.requires_grad = False
                for p in self.model.rpn.head.iou_pred.parameters():
                    p.requires_grad = False
                for p in self.model.rpn.head.bbox_pred.parameters():
                    p.requires_grad = False

            #if cfg_model.log_rpn:
            RegionProposalNetwork.forward = rpn_forward_that_save_infos_to_be_log
            RegionProposalNetwork.filter_proposals = filter_proposals_that_save_infos_to_be_log
            rpn_batch_size_per_image = 256
            self.model.rpn.batch_size_per_image = rpn_batch_size_per_image
            self.model.rpn.percent_of_negative_sample = cfg_model.percent_of_negative_sample 
            self.model.rpn.add_best_iou_sample = cfg_model.add_best_iou_sample


            if cfg_model.ai_oro:
                print("\nChange objectness from basic classification to iou regression and oro\n")

                GeneralizedRCNN.forward = my_rcnn_forward_with_oro
                RegionProposalNetwork.filter_proposals = filter_proposals_iou_with_oro
                RegionProposalNetwork.assign_targets_to_anchors = assign_targets_to_anchors_iou_oro
                if cfg_model.negative_sample:
                    RegionProposalNetwork.compute_loss = compute_loss_oro_iou_for_objecteness_with_negative_sample
                else:
                    RegionProposalNetwork.compute_loss = compute_loss_oro_iou_for_objecteness
                RegionProposalNetwork.forward = rpn_forward_oro_that_save_infos_to_be_log

                if not cfg_model.load:
                    out_channels = self.model.backbone.out_channels
                    rpn_anchor_generator = _default_anchorgen()
                    rpn_head = RPNHead_IoU_ORO(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
                    self.model.rpn.head = rpn_head

            elif cfg_model.ai_iou: 

                print("\nChange objectness from basic classification to iou regression\n")

                RegionProposalNetwork.compute_loss = compute_loss_iou_for_objecteness
                RegionProposalNetwork.assign_targets_to_anchors = assign_targets_to_anchors_iou
                RegionProposalNetwork.filter_proposals = filter_proposals_iou
                RegionProposalNetwork.forward = rpn_forward_that_save_infos_to_be_log


            self.model.rpn._pre_nms_top_n["training"] = cfg_model.rpn_pre_nms_training_top_n #2000
            self.model.rpn._pre_nms_top_n["testing"] = cfg_model.rpn_pre_nms_testing_top_n#1000
            self.model.rpn._post_nms_top_n["training"] = cfg_model.rpn_post_nms_training_top_n#2000
            self.model.rpn._post_nms_top_n["testing"] = cfg_model.rpn_post_nms_testing_top_n#1000

            self.model.roi_heads.unknown_roi_head_background_classif_score_threshold = cfg_model.unknown_roi_head_background_classif_score_threshold 
            self.model.roi_heads.unknown_roi_head_oro_score_threshold = cfg_model.unknown_roi_head_oro_score_threshold 
            self.model.roi_heads.unknown_roi_head_iou_score_threshold = cfg_model.unknown_roi_head_iou_score_threshold 
            self.oro_score_threshold = cfg_model.unknown_roi_head_oro_score_threshold


        # Yolov5
        elif cfg_model.name == "yolov5":
            print("Using yolov5 model ")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Yolov8
        elif cfg_model.name == "yolov8":
            print("Using yolov8 model ")
            self.model = YOLO('yolov8n.pt')
            self.dataset_path = cfg_model.dataset_path

        # Yolop
        elif cfg_model.name == "yolop":
            print("Using yolop model ")
            if cfg_model.pretrained and not cfg_model.load:
                self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
            else:
                raise NotImplementedError("This function is not yet implemented")

        else:
            print("[ERROR] Model name : " + cfg_model.name + " is not handle !")
            return 

        self.model.nb_classes = self.num_classes

        self.batch_size = batch_size
        print("Batch size :", batch_size)
        self.log("batch size", batch_size, batch_size=batch_size)
        self.show_image=show_image
        self.optimizer = cfg_model.optimizer
        self.learning_rate = cfg_model.learning_rate
        self.momentum = cfg_model.momentum
        self.weight_decay = cfg_model.weight_decay
        self.classes_names = classes_names_pred
        self.classes_names_gt = classes_names_gt
        self.classes_names_merged = classes_names_merged
        self.calcul_metric_train_each_nb_batch = cfg_model.calcul_metric_train_each_nb_batch
        self.save_each_epoch = cfg_model.save_each_epoch
        self.enable_unknown = cfg_model.enable_unknown
        self.nms_iou_threshold = cfg_model.nms_iou_threshold


        self.index = 0
        self.config = cfg_model
        self.model_name = cfg_model.name

        # Build dict with label associate to index for wandb
        self.class_id_label = {}
        for i, class_name in enumerate(self.classes_names):
            self.class_id_label[i] = class_name
        self.class_id_label[len(self.classes_names)] = "unknown"

        self.gt_class_id_label = {}
        for i, gt_class_name in enumerate(self.classes_names_gt):
            self.gt_class_id_label[i] = gt_class_name
        print("gt class", self.gt_class_id_label)

        self.merged_class_id_label = {}
        for i, merged_class_name in enumerate(self.classes_names_merged):
            self.merged_class_id_label[i] = merged_class_name
        print("merged id class", self.gt_class_id_label)

        self.semantic_segmentation_class_id_label = semantic_segmentation_class_id_label 

        wandb_alias = "offline" if wandb.run.name == None else wandb.run.name
        self.save_path = cfg_model.save_path + wandb_alias + "_" + cfg_model.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load(self, model_name):
        print("\nLoad model : ", model_name)
        self.model = torch.load(self.load_path + model_name, map_location=self.device)
        print("End Loading model\n")

    def forward(self, x):
        return self.model(x)

    def sort_predictions(self, predictions, filter_with_oro=False, nms_unknown_inside_known=True):

        # Select only known
        background_predictions = []
        known_predictions = []
        unknown_predictions = []
        for prediction in predictions:
            nb_pred = len(prediction["labels"])

            known_prediction = {}
            unknown_prediction = {}
            background_prediction = {}


            #filter out only those predicted labels that are recognized and considered valid based on the known classes you have defined.
            over_known_threshold_score_mask = prediction["scores"] >= self.scores_cfg.threshold_score 
            under_minimum_score_mask = prediction["scores"] < self.scores_cfg.threshold_score_minimum 
            classes_known_mask = torch.tensor([label in self.classes_known for label in prediction["labels"]], dtype=torch.bool, device=prediction["labels"].device)
            classes_background_mask = prediction["labels"] == 0

            # Mask with scores and classes
            if self.config.open_low_known_classes_as_unknown:
                background_mask = under_minimum_score_mask
            else:
                background_mask = under_minimum_score_mask | (~classes_background_mask & ~over_known_threshold_score_mask)  # If removing known classes between 0.2 and 0.5
            known_mask = classes_known_mask & over_known_threshold_score_mask
            unknown_mask = ~(background_mask | known_mask)

            if self.config.filter_with_oro:
                # Put unknown that have oro < threshold to background 
                unknown_with_oro_too_low_mask = unknown_mask & (prediction["oro"] < self.oro_score_threshold) #TODO make oro threshold parametable

                #update mask
                unknown_mask = unknown_mask ^ unknown_with_oro_too_low_mask
                background_mask = background_mask | unknown_with_oro_too_low_mask


            if nms_unknown_inside_known and unknown_mask.any() and known_mask.any():
                inter, unions = _box_inter_union(prediction["boxes"][unknown_mask], prediction["boxes"][known_mask])
                area_unknown = box_area(prediction["boxes"][unknown_mask])
                max_inter_values, max_inter_inds = inter.max(dim=1)

                unknown_inside_known_mask = ((area_unknown * 0.4).int() <= (max_inter_values).int())
                if unknown_inside_known_mask.any():
                    background_mask[unknown_mask] = unknown_inside_known_mask
                    unknown_mask[unknown_mask.clone()] = ~unknown_inside_known_mask

            nms_iou_threshold = 0.5  #TODO make in config iou nms thrshold
            kept_boxes_from_nms = torchvision.ops.nms(prediction["boxes"][unknown_mask], prediction['scores'][unknown_mask], nms_iou_threshold)  # NMS
            if kept_boxes_from_nms.any():
                mask_kept_unknown = torch.zeros(unknown_mask.sum(), dtype=bool, device=unknown_mask.device)
                mask_kept_unknown[kept_boxes_from_nms] = True
                background_mask[unknown_mask] = ~mask_kept_unknown
                unknown_mask[unknown_mask.clone()] = mask_kept_unknown 

            for key, value in prediction.items():

                if not torch.is_tensor(value):
                    continue

                known_prediction[key] = value[known_mask]
                unknown_prediction[key] = value[unknown_mask]
                background_prediction[key] = value[background_mask]

            sum_pred = len(known_prediction["labels"]) + len(background_prediction["labels"]) + len(unknown_prediction["labels"])
            if nb_pred != sum_pred:
                print(f"[ERROR] Sorting between K, U and B gone wrong :( sum prediction ({sum_pred}) != nb pred ({nb_pred})")
                print("\n\nknown : ", known_prediction)
                print("\n\nunknown : ",unknown_prediction)
                print("\n\nBackground : ",background_prediction)
                print(f"[ERROR] Sorting between K, U and B gone wrong :( sum prediction ({sum_pred}) != nb pred ({nb_pred})")
                exit()


            known_predictions.append(known_prediction)
            unknown_predictions.append(unknown_prediction)
            background_predictions.append(background_prediction)




        return known_predictions, unknown_predictions, background_predictions

    def get_predictions(self, images, targets, batch_idx, training=False):

        if "predictions" in targets[0]:
            predictions = []
            for target in targets:
                predictions.append(target["predictions"])
        else:
            if self.model_name == "yolov8":

                images_names = []
                for i, image in enumerate(images):
                    name = targets[i]["name"]
                    zeros_needed = 12 - len(name)
                    name = self.dataset_path + "images/" + "0" * zeros_needed + name + ".jpg"
                    images_names.append(name)
                
                predictions = self.model(images_names, verbose=False)
                predictions = convert_yolo_prediction_format(predictions)

            elif self.model_name == "yolop":
                predictions, da_seg_out, ll_seg_out = self.model(images)
                inference_output, train_out = predictions

                # TODO change hardcode value below !!!!
                predictions = non_max_suppression(inference_output, conf_thres=0.2, iou_thres=0.5)
                predictions = convert_yolop_prediction_format(predictions)

            elif self.model_name == "faster_rcnn":

                predictions = self.model(images)

                if self.config.ai_oro: 
                    for index in range(len(predictions)):
                        predictions[index]["oro"] = self.model.roi_heads.all_oro_scores[index]
                        predictions[index]["iou"] = self.model.roi_heads.all_iou_scores[index]

                batch_max_for_log = 2
                if batch_idx <= batch_max_for_log or self.config.prediction_from_rpn:

                    transform_images, transform_targets = self.model.transform(images, targets)

                    original_image_sizes: List[Tuple[int, int]] = []
                    for img in images:
                        val = img.shape[-2:]
                        torch._assert(
                                len(val) == 2,
                                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                                )
                        original_image_sizes.append((val[0], val[1]))

                    if batch_idx <= batch_max_for_log and self.config.log_rpn and not self.config.prediction_from_rpn:
                        proposals_detections , anchors_detections = self.get_rpn_detections(transform_images, original_image_sizes)
                        self.log_rpn_proposals(proposals_detections, anchors_detections)

                        if self.config.ai_oro: 
                            self.log_rpn_sampling(transform_images, original_image_sizes, transform_targets)

                if self.config.prediction_from_rpn:
                    proposals_detections, anchors_detections = self.get_rpn_detections(transform_images, original_image_sizes)
                    predictions = proposals_detections 
            else:
                predictions = self.model(images)

            if self.save_predictions:
                self.save_predictions_json(predictions, targets)

        return predictions

    def log_rpn_sampling(self, transform_images, original_image_sizes, transform_targets):

        sampled_anchors = self.model.rpn.current_sampled_anchors
        sampled_iou_targets = self.model.rpn.current_sampled_iou_targets
        sampled_iou_predictions = self.model.rpn.current_sampled_iou_predictions

        # GT 
        log_targets = []
        for target in transform_targets:
            log_targets.append({"boxes": target["boxes"], "labels": torch.zeros(len(target["boxes"]), device=target["boxes"].device, dtype=int)})

        # objectness target
        anchors_target = []
        anchors_detection = []
        for anchor, target, prediction in zip(sampled_anchors, sampled_iou_targets, sampled_iou_predictions):
            anchors_target.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": target})
            anchors_detection.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": prediction})

        anchors_detections = self.model.transform.postprocess(anchors_detection, transform_images.image_sizes, original_image_sizes)
        anchors_target = self.model.transform.postprocess(anchors_target, transform_images.image_sizes, original_image_sizes)

        wandb_rpn_sampling_target = []
        wandb_rpn_sampling_predictions = []
        for i, image in enumerate(transform_images.tensors):
            wandb_rpn_sampling_target.append(get_wandb_image_with_labels(image, log_targets[i], anchors_target[i], {0: "target"}, {0: "gt"}))
            wandb_rpn_sampling_predictions.append(get_wandb_image_with_labels(image, log_targets[i], anchors_detections[i], {0: "prediction"}, {0: "gt"}))

        wandb.log({("Images/rpn_sampling_target"): wandb_rpn_sampling_target})
        wandb.log({("Images/rpn_sampling_predictions"): wandb_rpn_sampling_predictions})

        if self.config.ai_oro: 

            sampled_oro_targets = self.model.rpn.current_oro_targets 
            sampled_oro_predictions = self.model.rpn.current_oro_predictions
            oro_sampled_anchors = self.model.rpn.current_oro_sampled_anchors

            anchors_target_oro = []
            anchors_detection_oro = []
            for anchor, target, prediction in zip(oro_sampled_anchors, sampled_oro_targets, sampled_oro_predictions):
                anchors_target_oro.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": target})
                anchors_detection_oro.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": prediction})

            anchors_detections_oro = self.model.transform.postprocess(anchors_detection_oro, transform_images.image_sizes, original_image_sizes)
            anchors_target_oro = self.model.transform.postprocess(anchors_target_oro, transform_images.image_sizes, original_image_sizes)

            wandb_rpn_sampling_target_oro = []
            wandb_rpn_sampling_predictions_oro = []
            for i, image in enumerate(transform_images.tensors):
                wandb_rpn_sampling_target_oro.append(get_wandb_image_with_labels(image, log_targets[i], anchors_target_oro[i], {0: "target"}, {0: "gt"}))
                wandb_rpn_sampling_predictions_oro.append(get_wandb_image_with_labels(image, log_targets[i], anchors_detections_oro[i], {0: "prediction"}, {0: "gt"}))

            wandb.log({("Images/rpn_sampling_target_oro"): wandb_rpn_sampling_target_oro})
            wandb.log({("Images/rpn_sampling_predictions_oro"): wandb_rpn_sampling_predictions_oro})

    def get_rpn_detections(self, transform_images, original_image_sizes):

        proposals = self.model.rpn.current_proposals
        proposal_scores = self.model.rpn.current_scores
        anchors = self.model.rpn.current_anchors
        anchors = self.model.rpn.current_filtered_anchors

        proposals_detections = []
        for proposal, proposal_score in zip(proposals, proposal_scores):
            proposals_detections.append({"boxes": proposal, "labels": torch.zeros(len(proposal), device=proposal.device, dtype=int), "scores": proposal_score})
        proposals_detections = self.model.transform.postprocess(proposals_detections, transform_images.image_sizes, original_image_sizes)

        anchors_detections = []
        for anchor, proposal_score in zip(anchors, proposal_scores): #anchors_labels):
            anchors_detections.append({"boxes": anchor, "labels": torch.zeros(len(anchor), device=anchor.device, dtype=int), "scores": proposal_score})
        anchors_detections = self.model.transform.postprocess(anchors_detections, transform_images.image_sizes, original_image_sizes)

        return proposals_detections, anchors_detections


    def log_rpn_proposals(self, proposals_detections, anchors_detections):

        # WANDB LOG object proposal
        wandb_object_proposal_images = []
        for i, image in enumerate(transform_images.tensors):
            wandb_object_proposal_images.append(get_wandb_image_with_labels(image, proposals_detections[i], anchors_detections[i], {0: "anchors"}, {0: "proposals"}))

        wandb.log({("Images/Objects_proposal"): wandb_object_proposal_images})


    # ------ TRAINNING ----------------------------------


    def training_step(self, batch, batch_idx):

        images, targets = batch

        for target in targets:
            if len(target["boxes"]) == 0:
                #print("SKIP BATCH BECAUSE EMPTY TARGETS !!! :", target)
                return None
            if (target["boxes"][:, 0] >= target["boxes"][:, 2]).any() or (target["boxes"][:, 1] >= target["boxes"][:, 3]).any():
                return None


        # Remove class we can't predict:
        targets = get_only_known_targets(targets)

        if self.model_name == "fcos":
            (losses, predictions) = self.model(images, targets)

        elif self.model_name == "retina_net":
            # Forward + Loss 
            loss_dict = self.model(images, targets)
            self.log("losses", loss_dict)
            losses = sum(loss for loss in loss_dict.values())

        elif self.model_name == "faster_rcnn":

            # Apply transform on the SS OBD and get transfrom img for logging
            if self.config.ai_oro or self.config.perfect_oro: 
                for target in targets:
                    target["masks"] = target["semantic_segmentation_OBD"].unsqueeze(0)

            transform_images, transform_targets = self.model.transform(images, targets)

            if self.config.ai_oro or self.config.perfect_oro: 
                for target, transform_target in zip(targets, transform_targets):
                    target["semantic_segmentation_OBD"] = transform_target["masks"].squeeze(0)


            # Forward + Loss 
            loss_dict = self.model(images, targets)
            self.log("losses", loss_dict)
            losses = sum(loss for loss in loss_dict.values())

            # log 
            if batch_idx <= 2: # or batch_idx == 2:

                original_image_sizes: List[Tuple[int, int]] = []
                for img in images:
                    val = img.shape[-2:]
                    torch._assert(
                            len(val) == 2,
                            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
                            )
                    original_image_sizes.append((val[0], val[1]))

                proposals_detections , anchors_detections = self.get_rpn_detections(transform_images, original_image_sizes)
                self.log_rpn_proposals(proposals_detections, anchors_detections)
                self.log_rpn_sampling(transform_images, original_image_sizes, transform_targets)

        else:
            raise Exception("not implemented")

        if self.current_epoch % self.save_each_epoch == 0 and self.current_epoch != 0:
            path = self.save_path + "epoch_" + str(self.current_epoch)
            torch.save(self.model, path)

        return losses

    def on_train_end(self):

        torch.save(self.model, self.save_path + "Final")
    
    # ------ Evaluation ----------------------------------

    def evaluation_step(self, batch, batch_idx, metrics_module, loging_name, with_randoms=False):


        images, targets = batch
        predictions = self.get_predictions(images, targets, batch_idx)

        # Add all postprocess information needed by states (scores, tags, areas, etc..)
        with_perfect_oro = self.config.perfect_oro
        customs_scores_concatenate_randoms, customs_scores_concatenate_true_randoms = postprocess_predictions(images, predictions, targets, self.scores_cfg, random=with_randoms, with_perfect_oro=with_perfect_oro)
        #if with_randoms and self.batch_size > 1:
        #    metrics_module.update_randoms(customs_scores_concatenate_randoms, customs_scores_concatenate_true_randoms)

        # Seperate targets and predictions between known and unknown 
        known_targets = get_only_known_targets(targets)
        unknown_targets = get_only_unknown_targets(targets)
        background_targets = get_only_background_targets(targets)

        known_predictions, unknown_predictions, background_predictions = self.sort_predictions(predictions, filter_with_oro=self.config.filter_with_oro)

        # Match predictions and targets
        for batch_index in range(len(targets)):

            set_tags(known_predictions, known_targets, batch_index, self.scores_cfg.iou_threshold, self.considering_known_classes)
            set_tags(unknown_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False)
            set_tags(known_predictions, unknown_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_KP_with_UT")
            #set_tags(unknown_predictions, known_targets, batch_index, self.scores_cfg.iou_threshold, considering_classes=False, tags_name="tags_UP_with_KT")

        # Update with predictions 
        metrics_module.update(known_targets, unknown_targets, background_targets, known_predictions, unknown_predictions, background_predictions, targets)


        # Loging process
        target_known_unknown_and_background = []
        target_and_random_wandb_images = []
        edge_wandb_images = []

        log_A_OSE = False #TODO put it in config file
        if log_A_OSE and batch_idx <= 2: # or batch_idx == 2:
            A_OSE_wandb_images = []
            for i in range(len(images)):

                if known_predictions[i]["tags_KP_with_UT"].any():

                    A_OSE_wandb_images.append(get_wandb_image_with_labels_background_unknown_known(images[i],
                                                                                                (known_targets[i], unknown_targets[i]),
                                                                                                (known_predictions[i], unknown_predictions[i], background_predictions[i]),
                                                                                                pred_class_id_label=self.class_id_label,
                                                                                                gt_class_id_label=self.gt_class_id_label,
                                                                                                semantic_segmentation_class_id_label=self.semantic_segmentation_class_id_label))

            if A_OSE_wandb_images != []:
                wandb.log({("Images/" + loging_name + "_A_OSE"): A_OSE_wandb_images})

        # If first val, log images and pred
        if batch_idx <= 2: # or batch_idx == 2:
            nn_wandb_images = []
            for i in range(len(images)):

                nn_wandb_images.append(get_wandb_image_with_labels_background_unknown_known(images[i],
                                                                                            (known_targets[i], unknown_targets[i]),
                                                                                            (known_predictions[i], unknown_predictions[i], background_predictions[i]),
                                                                                            pred_class_id_label=self.class_id_label,
                                                                                            gt_class_id_label=self.merged_class_id_label,
                                                                                            semantic_segmentation_class_id_label=self.semantic_segmentation_class_id_label,
                                                                                            display=False, img_id=str(batch_idx) + str(i)))

                """
                if self.batch_size > 1:
                    #print(len(images), len(targets), len(customs_scores_concatenate_randoms), customs_scores_concatenate_randoms, len(customs_scores_concatenate_randoms["boxes"]),  i)
                    target_and_random_wandb_images.append(get_wandb_image_with_labels_target_background(images[i], targets[i], customs_scores_concatenate_randoms, i,
                                                                                            pred_class_id_label=self.class_id_label,
                                                                                            gt_class_id_label=self.gt_class_id_label))
                """

                target_known_unknown_and_background.append(get_wandb_image_with_labels_target_known_unknown_background(images[i], known_targets[i], unknown_targets[i], background_targets[i],
                                                                                            pred_class_id_label=self.class_id_label,
                                                                                            gt_class_id_label=self.merged_class_id_label))


            wandb.log({("Images/" + loging_name + "_nn"): nn_wandb_images})

            if batch_idx <= -1: # or batch_idx == 2:
                if self.batch_size > 1:
                    wandb.log({("Images/" + loging_name + "_target_vs_randoms"): target_and_random_wandb_images})
                wandb.log({("Images/" + loging_name + "_target_known_unknown_background"): target_known_unknown_and_background})


    def plot_kubt(self, known_predictions, unknown_predictions, background_predictions, targets):

        fig, ax = plt.subplots()

        for i in range(len(known_predictions)):

            """
            ax.scatter(known_predictions[i]["scores"].cpu().numpy(), known_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:blue", label="Known")
            ax.scatter(unknown_predictions[i]["scores"].cpu().numpy(), unknown_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:red", label="Unknown")
            ax.scatter(background_predictions[i]["scores"].cpu().numpy(), background_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:green", label="Background")
            """

            ax.scatter(known_predictions[i]["custom_scores"]["edge_density"].cpu().numpy(), known_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:blue", label="Known")
            ax.scatter(unknown_predictions[i]["custom_scores"]["edge_density"].cpu().numpy(), unknown_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:red", label="Unknown")
            ax.scatter(background_predictions[i]["custom_scores"]["edge_density"].cpu().numpy(), background_predictions[i]["custom_scores"]["color_contrast"].cpu().numpy(), c="tab:green", label="Background")

        ax.legend()
        ax.grid(True)


    # ------ Validation ----------------------------------

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        for target in targets:
            if len(target["boxes"]) == 0:
                return None
            if (target["boxes"][:, 0] >= target["boxes"][:, 2]).any() or (target["boxes"][:, 1] >= target["boxes"][:, 3]).any():
                return None
        self.evaluation_step(batch, batch_idx, self.val_metrics, "Validation")

    def validation_epoch_end(self, validation_step_outputs):

        wandb.log({"epoch": self.current_epoch})

        # Save model
        if self.current_epoch % self.save_each_epoch == 0 and self.current_epoch != 0:
            path = self.save_path + "epoch_" + str(self.current_epoch)
            torch.save(self.model, path)

        wandb.log({"Val/metrics": self.val_metrics.get_wandb_metrics(with_print=True)})

        val_map_known = self.val_metrics.current_known_map['map']
        self.log("val_map_known", val_map_known)
        self.log("val_map_unknown", self.val_metrics.current_unknown_map['map'])

        if val_map_known >= self.best_map:
            torch.save(self.model, self.save_path + "epoch_" + str(self.current_epoch) + "Best_map_" + str(val_map_known.item()))
            self.best_map = val_map_known


    # ------ TESTING ----------------------------------

    def test_step(self, batch, batch_idx):

        images, targets = batch
        for target in targets:
            if len(target["boxes"]) == 0:
                return None
            if (target["boxes"][:, 0] >= target["boxes"][:, 2]).any() or (target["boxes"][:, 1] >= target["boxes"][:, 3]).any():
                return None

        self.evaluation_step(batch, batch_idx, self.test_metrics, "Test")

        return 0 


    def test_epoch_end(self, validation_step_outputs):

        wandb.log({"Test/metrics": self.test_metrics.get_wandb_metrics(with_print=True)})

        test_map_known = self.test_metrics.current_known_map['map']
        self.log("test_map_known", test_map_known)
        self.log("test_precision_known", self.test_metrics.get_known_precision())
        self.log("test_recall_known", self.test_metrics.get_known_recall())
        self.log("test_map_unknown", self.test_metrics.current_unknown_map['map'])
        self.log("test_precision_unknown", self.test_metrics.get_unknown_precision())
        self.log("test_recall_unknown", self.test_metrics.get_unknown_recall())
        self.log("test_A-OSE", self.test_metrics.get_open_set_errors())

    
    # -------------- MISC -------------------------------------------------------

    def save_predictions_json(self, predictions, targets):
        
        for i, prediction in enumerate(predictions):
            image_name = targets[i]["name"]
            json_file_path = self.save_predictions_path + self.model_name + "/" + image_name + ".json"
            print("Writing predictions in :", json_file_path)

            json_prediction = {
                                "boxes": prediction["boxes"].tolist(),
                                "labels": prediction["labels"].tolist(),
                                "scores": prediction["scores"].tolist(),
                              }

            json_target = {
                            "boxes": targets[i]["boxes"].tolist(),
                            "labels": targets[i]["labels"].tolist(),
                          }


            with open(json_file_path, "w") as json_file:
                data = {"image_name": image_name, "model_name": self.model_name, "predictions": json_prediction, "targets": json_target, "labels_id_names": self.gt_class_id_label}
                json.dump(data, json_file, indent=4)


    def configure_optimizers(self):

        if self.optimizer == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

        if self.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    # Gives a mask of anchor points with their centerness classes
    def get_prediction_masks_fcos(self, image_size, batch_size, interval=10):
        masks = {}

        # Mask of centerness
        if self.enable_semantics_centerness:
            foregroud_mask = self.model.head.centerness_foregroud_mask[batch_size - 1]
        else:
            foregroud_mask = self.model.head.foregroud_mask[batch_size - 1]

        pred_centerness = (self.model.head.pred_centerness[batch_size - 1][foregroud_mask]) #[N, HWA]
        gt_centerness = self.model.head.gt_ctrness_targets[batch_size - 1][foregroud_mask]
        anchor_centers = self.model.anchor_centers[foregroud_mask]

        gt_centerness_mask = torch.zeros(image_size).int()
        pred_centerness_mask = torch.zeros(image_size).int()

        for i in range(0, 100, interval):
            all_anchor_in_interval_pred = (pred_centerness >= i/100) & (pred_centerness < (i/100 + interval))
            all_anchor_in_interval_gt = (gt_centerness >= i/100) & (gt_centerness < (i/100 + interval))
            for b in range(-2, 2):
                for j in range(-2, 2):
                    pred_centerness_mask[anchor_centers[all_anchor_in_interval_pred, 1].long() + b, anchor_centers[all_anchor_in_interval_pred, 0].long() + j] = i + interval
                    gt_centerness_mask[anchor_centers[all_anchor_in_interval_gt, 1].long() + b, anchor_centers[all_anchor_in_interval_gt, 0].long() + j] = i + interval

        masks["centerness prediction"] = {"mask_data": pred_centerness_mask.cpu().numpy()}
        masks["centerness ground_truth"] = {"mask_data": gt_centerness_mask.cpu().numpy()}

        return masks

    def get_image_with_fcos_predictions(self, last_image, batch_size):

        image_size = (last_image.shape[1], last_image.shape[2])

        anchor_centers = self.model.anchor_centers
        gt_centers = self.model.gt_centers 
        last_targets = self.model.last_targets
        last_pred = self.model.head.last_pred_boxes
        gt_classes_targets = self.model.head.gt_classes_targets[batch_size - 1]
        cls_logits = self.model.head.cls_logits[batch_size - 1] 
        num_anchors_per_level = self.model.num_anchors_per_level

        name = last_targets["name"]

        #centerness
        pred_centerness = self.model.head.pred_centerness[batch_size - 1].detach()
        gt_centerness = torch.nan_to_num(self.model.head.gt_ctrness_targets[batch_size - 1])

        if self.enable_semantics_centerness:
            foregroud_mask = self.model.head.centerness_foregroud_mask[batch_size - 1]
        else:
            foregroud_mask = self.model.head.foregroud_mask[batch_size - 1]

        gt_box_data = self.get_wandb_box(last_targets, image_size)
        pred_box_data = self.get_wandb_box(last_pred[batch_size -1], image_size)
        boxes={"ground_truth": {"box_data": gt_box_data, "class_labels": self.class_id_label},
               "predictions": {"box_data": pred_box_data, "class_labels": self.class_id_label}}

        levels = [8, 16, 32, 64, 128]
        masks = {}

        anchors_indexes = 0
        for i, level in enumerate(levels):
            if "seg_masks" in last_targets:
                seg_gt_mask = torch.zeros(image_size, device=last_targets["boxes"].device).long()

            centerness_pred_mask = torch.zeros(image_size, device=last_targets["boxes"].device, dtype=pred_centerness.dtype)
            classes_pred_masks = [torch.zeros(image_size, device=last_targets["boxes"].device).int() for i in range(self.model.nb_classes)]

            anchors_indexes_next = anchors_indexes + int(((last_image.shape[2]/level) * last_image.shape[1]/level))
            anchor_centers_x = anchor_centers[anchors_indexes:anchors_indexes_next, 1].long()
            anchor_centers_y = anchor_centers[anchors_indexes:anchors_indexes_next, 0].long()

            for b in range(-2, 2):
                for j in range(-2, 2):

                    nothing_mask = torch.sum(gt_classes_targets[anchors_indexes:anchors_indexes_next], dim=1) == 0

                    if "seg_masks" in last_targets:
                        seg_gt_mask[anchor_centers_x + b, anchor_centers_y + j] = 254 * nothing_mask
                        seg_gt_mask[anchor_centers_x + b,anchor_centers_y  + j] += torch.argmax(gt_classes_targets[anchors_indexes:anchors_indexes_next], dim=1) + 1
                        seg_gt_mask[gt_centers[:, 1].long() + b, gt_centers[:, 0].long() + j] = len(self.classes_names) + 1

                    centerness_pred_mask[anchor_centers_x + b, anchor_centers_y + j] = pred_centerness[anchors_indexes:anchors_indexes_next]

                    for c in range(self.model.nb_classes):
                        classes_pred_masks[c][anchor_centers_x + b, anchor_centers_y + j] = (cls_logits[anchors_indexes:anchors_indexes_next, c] >= 0.5).int() #[H, W] <- [HWA, 11] 


            anchors_indexes = anchors_indexes_next
            if "seg_masks" in last_targets:
                masks['ground_truth_class_level_' + str(i)] = {"mask_data": seg_gt_mask.cpu().numpy(), "class_labels": self.class_labels}
            masks['prediction_centerness_level_' + str(i)] = {"mask_data": centerness_pred_mask.cpu().numpy()}

        masks.update(self.get_prediction_masks_fcos(image_size, batch_size))

        return wandb.Image(last_image.cpu(), masks=masks, boxes=boxes)

    def log_fcos_heatmaps(self, last_image, batch_size):

        image_size = (last_image.shape[1], last_image.shape[2])

        anchor_centers = self.model.anchor_centers
        gt_centers = self.model.gt_centers 
        last_targets = self.model.last_targets
        last_pred = self.model.head.last_pred_boxes
        gt_classes_targets = self.model.head.gt_classes_targets[batch_size - 1]
        cls_logits = self.model.head.cls_logits[batch_size - 1] 
        num_anchors_per_level = self.model.num_anchors_per_level

        name = last_targets["name"]

        #centerness
        pred_centerness = self.model.head.pred_centerness[batch_size - 1].detach()
        gt_centerness = torch.nan_to_num(self.model.head.gt_ctrness_targets[batch_size - 1])

        if self.enable_semantics_centerness:
            foregroud_mask = self.model.head.centerness_foregroud_mask[batch_size - 1]
        else:
            foregroud_mask = self.model.head.foregroud_mask[batch_size - 1]

        levels = [8, 16, 32, 64, 128]

        anchors_indexes = 0
        for i, level in enumerate(levels):

            centerness_pred_mask = torch.zeros(image_size, device=last_targets["boxes"].device, dtype=pred_centerness.dtype)
            classes_pred_masks = [torch.zeros(image_size, device=last_targets["boxes"].device).int() for i in range(self.model.nb_classes)]

            centerness_pred_heatmap = torch.zeros_like(centerness_pred_mask)
            centerness_gt_heatmap = torch.zeros_like(centerness_pred_mask, dtype=gt_centerness.dtype)
            centerness_gt_heatmap_foregroud = torch.zeros_like(centerness_gt_heatmap)
            classes_pred_heatmaps = [torch.zeros_like(centerness_pred_mask) for i in range(self.model.nb_classes)]

            anchors_indexes_next = anchors_indexes + int(((last_image.shape[2]/level) * last_image.shape[1]/level))
            anchor_centers_x = anchor_centers[anchors_indexes:anchors_indexes_next, 1].long()
            anchor_centers_y = anchor_centers[anchors_indexes:anchors_indexes_next, 0].long()

            for b in range(-2, 2):
                for j in range(-2, 2):

                    centerness_pred_heatmap[anchor_centers_x + b, anchor_centers_y + j] = pred_centerness[anchors_indexes:anchors_indexes_next]
                    centerness_gt_heatmap[anchor_centers_x + b, anchor_centers_y + j] = gt_centerness[anchors_indexes:anchors_indexes_next]
                    centerness_gt_heatmap_foregroud[anchor_centers_x + b, anchor_centers_y + j] = gt_centerness[anchors_indexes:anchors_indexes_next] * foregroud_mask[anchors_indexes:anchors_indexes_next].int()
                    for c in range(self.model.nb_classes):
                        classes_pred_heatmaps[c][anchor_centers_x + b, anchor_centers_y + j] = cls_logits[anchors_indexes:anchors_indexes_next, c] #[H, W] <- [HWA, 11] 

            anchors_indexes = anchors_indexes_next

            if i == 0:
                for c in range(self.model.nb_classes):
                    if not torch.all(classes_pred_heatmaps[c] == 0):
                        image_heatmap = np.stack((classes_pred_heatmaps[c].detach().cpu().numpy(),)*3, axis=-1)
                        wandb.log({("Images/inside_train/heatmap/prediction_class_" + self.class_labels[c + 1] + "_level_" + str(i)) : wandb.Image(image_heatmap)}, commit=False)

            image_heatmap = np.stack((centerness_pred_heatmap.cpu().detach().numpy(),)*3, axis=-1)
            wandb.log({("Images/inside_train/heatmap/prediction_centerness_level_" + str(i)) : wandb.Image(image_heatmap)}, commit=False)
            if not torch.all(classes_pred_heatmaps[c] == 0):
                image_heatmap = np.stack((centerness_gt_heatmap_foregroud.cpu().numpy(),)*3, axis=-1)
                wandb.log({("Images/inside_train/heatmap/gt_centerness_foregroud_level_" + str(i)) : wandb.Image(image_heatmap)}, commit=False)

        wandb.log({("Images/inside_train/heatmap/image") : wandb.Image(last_image.cpu()), "Image/inside_train/heatmap/image_name": name})


