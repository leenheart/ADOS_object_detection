import wandb
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Logger():

    def __init__(self, dataset_class_names, model_class_names):

        self.dataset_classes_names = dataset_class_names
        self.dataset_classes_names_dict = {}
        for i, classe in enumerate(self.dataset_classes_names):
            self.dataset_classes_names_dict[i] = classe

        self.model_classes_names = model_class_names
        self.model_classes_names_dict = {}
        for i, classe in enumerate(self.model_classes_names):
            self.dataset_classes_names_dict[i] = classe

        self.model_classes_names_with_unknown = model_class_names + "unknown"
        self.model_classes_names_dict_with_unknown = {}
        for i, classe in enumerate(self.model_classes_names_with_unknown):
            self.model_classes_names_dict_with_unknown[i] = classe

    def log_image_batch(self, images, targets, predictions, log_name="Images", unknown=False):

        if images == None:
            return

        if unknown:
            class_names_pred = self.model_classes_names_with_unknown 
            class_names_dict_pred = self.model_classes_names_dict_with_unknown
        else:
            class_names_pred = self.model_classes_names
            class_names_dict_pred = self.model_classes_names_dict

        wandb_images = []
        for i, image in enumerate(images):
            boxes = {}
            if targets != None:
                boxes["ground_truth"] = {"box_data": get_wandb_box(targets[i], self.dataset_classes_names, (image.shape[1], image.shape[2])), "class_labels": self.dataset_classes_names_dict}
            if predictions != None:
                boxes["predictions"] = {"box_data": get_wandb_box(predictions[i], class_names_pred, (image.shape[1], image.shape[2])), "class_labels": class_names_dict_pred}
            wandb_images.append(wandb.Image(image, boxes=boxes))


        wandb.log({"Images/" + log_name: wandb_images})

def get_wandb_box(targets, class_id_label, image_size):

    if not type(targets) is dict:
        print("WTF does not get a dict for wandb box targets !")
        return []

    #'boxes', 'labels', 'scores', 'oro', 'iou', 'tags', 'IoU'
    boxes = targets["boxes"].cpu()
    labels = targets["labels"].cpu()
    scores = targets["scores"].cpu() if "scores" in targets else None
    tags_open_set_errors = targets["tags_KP_with_UT"].cpu() if "tags_KP_with_UT" in targets else None
    ious = targets["IoU"].cpu() if "IoU" in targets else None
    pred_ious = targets["iou"].cpu() if "iou" in targets else None
    scores_classe = targets["scores_classe"].cpu() if "scores_classe" in targets else None
    scores_centerness = targets["scores_centerness"].cpu() if "scores_centerness" in targets else None
    scores_color_contrast = targets["custom_scores"]["color_contrast"] if "custom_scores" in targets else None
    scores_edge = targets["custom_scores"]["edge_density"] if "custom_scores" in targets else None
    scores_on_drivable = targets["custom_scores"]["object_on_drivable"] if ("custom_scores" in targets and "object_on_drivable" in targets["custom_scores"]) else None
    scores_semantic_segmentation = targets["score_semantic_segmentation"] if "score_semantic_segmentation" in targets else None
    scores_semantic_segmentation_drivable = targets["score_drivable_pourcent"] if "score_drivable_pourcent" in targets else None
    scores_oro = targets["oro"] if "oro" in targets else None

    image_x_size = image_size[1]
    image_y_size = image_size[0]

    boxes_wandb = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        class_id = labels[i].item()

        box_caption = class_id_label[class_id] + " ("
        if not tags_open_set_errors is None:
            box_caption += " A-OSE " if tags_open_set_errors[i].item() else ""
        if not scores is None:
            box_caption += str(round(scores[i].item(), 2))
        if not scores_on_drivable is None:
            box_caption += " OBD:" + str(round(scores_on_drivable[i].item(), 1))
        """
        if not scores_edge is None:
            box_caption += " CC:" + str(round(scores_color_contrast[i].item(), 1)) + " E:" + str(round(scores_edge[i].item(), 1))
        if not scores_semantic_segmentation is None:
            box_caption += " SS:" + str(round(scores_semantic_segmentation[i].item(), 1))
        if not scores_semantic_segmentation_drivable is None:
            box_caption += " SS_drivable:" + str(round(scores_semantic_segmentation_drivable[i].item(), 1))
        if not ious is None:
            box_caption += " Iou:" + str(round(ious[i].item(), 1))
        """
        if not scores_oro is None:
            box_caption += " ORO:" + str(round(scores_oro[i].item(), 1))
        if not pred_ious is None:
            box_caption += " p_iou:" + str(round(pred_ious[i].item(), 1))
        
        box_caption += ")" 

        boxes_wandb.append({"position":{
                                "minX": x1.item() / image_x_size,
                                "maxX": x2.item() / image_x_size,
                                "minY": y1.item() / image_y_size,
                                "maxY": y2.item() / image_y_size
                                },
                            "class_id": class_id,
                            "box_caption": box_caption
                            })
        if scores != None:
            boxes_wandb[i]["scores"] = {"objectness": scores[i].item()}
            if not scores_oro is None:
                boxes_wandb[i]["scores"]["scores_oro"] = scores_oro[i].item()
            if not pred_ious is None:
                boxes_wandb[i]["scores"]["scores_pred_iou"] = pred_ious[i].item()
            if scores_centerness != None:
                boxes_wandb[i]["scores"]["scores_centerness"] = scores_centerness[i].item()
            if scores_classe != None:
                boxes_wandb[i]["scores"]["scores_classe"] = torch.max(scores_classe[i]).item()
            if scores_color_contrast != None:
                boxes_wandb[i]["scores"]["scores_color_contrast"] = scores_color_contrast[i].item()
            if scores_edge!= None:
                boxes_wandb[i]["scores"]["scores_edge"] = scores_color_contrast[i].item()

    return boxes_wandb

def get_wandb_image_with_proposal(image, proposal, scores):

    proposal = {"boxes": proposal, "labels": torch.zeros(len(proposal), device=proposal.device, dtype=int), "scores": scores}
    gt_box_data = get_wandb_box(proposal, {0: ""}, (image.shape[1], image.shape[2]))
    boxes={"object_proposal": {"box_data": gt_box_data}}

    return wandb.Image(image, boxes=boxes)

def get_wandb_image_with_labels(image, targets, predictions, pred_class_id_label, gt_class_id_label=None):

    if gt_class_id_label == None:
        gt_class_id_label = pred_class_id_label

    gt_box_data = get_wandb_box(targets, gt_class_id_label, (image.shape[1], image.shape[2]))
    pred_box_data = get_wandb_box(predictions, pred_class_id_label, (image.shape[1], image.shape[2]))
    boxes={"ground_truth": {"box_data": gt_box_data, "class_labels": gt_class_id_label},
            "predictions": {"box_data": pred_box_data, "class_labels": pred_class_id_label}}

    masks = None
    #if "seg_masks" in targets:
    #    masks = {"ground_truth": {"mask_data": np.clip(targets["seg_masks"].cpu().numpy() + 1, 0, 255), "class_labels": self.class_labels}}

    return wandb.Image(image, boxes=boxes, masks=masks)


def get_wandb_image_with_labels_target_known_unknown_background(image, knowns, unknowns, backgrounds, pred_class_id_label, gt_class_id_label=None):

    if gt_class_id_label == None:
        gt_class_id_label = pred_class_id_label

    known_box_data = get_wandb_box(knowns, gt_class_id_label, (image.shape[1], image.shape[2]))
    unknown_box_data = get_wandb_box(unknowns, gt_class_id_label, (image.shape[1], image.shape[2]))
    background_box_data = get_wandb_box(backgrounds, gt_class_id_label, (image.shape[1], image.shape[2]))

    boxes={"knowns": {"box_data": known_box_data, "class_labels": gt_class_id_label},
            "unknowns": {"box_data": unknown_box_data, "class_labels": pred_class_id_label},
            "backgrounds": {"box_data": background_box_data, "class_labels": pred_class_id_label}}

    return wandb.Image(image, boxes=boxes)


def get_wandb_image_with_labels_background_unknown_known(images, targets, predictions, pred_class_id_label, gt_class_id_label=None, semantic_segmentation_class_id_label=None, display=False, img_id=""):

    known_targets, unknown_targets = targets
    known_predictions, unknown_predictions, background_predictions = predictions

    if gt_class_id_label == None:
        gt_class_id_label = pred_class_id_label

    known_gt_box_data = get_wandb_box(known_targets, gt_class_id_label, (images.shape[1], images.shape[2]))
    unknown_gt_box_data = get_wandb_box(unknown_targets, gt_class_id_label, (images.shape[1], images.shape[2]))
    known_pred_box_data = get_wandb_box(known_predictions, pred_class_id_label, (images.shape[1], images.shape[2]))
    unknown_pred_box_data = get_wandb_box(unknown_predictions, pred_class_id_label, (images.shape[1], images.shape[2]))
    background_pred_box_data = get_wandb_box(background_predictions, pred_class_id_label, (images.shape[1], images.shape[2]))

    boxes={"known ground truth": {"box_data": known_gt_box_data, "class_labels": gt_class_id_label},
            "unknown ground truth": {"box_data": unknown_gt_box_data, "class_labels": gt_class_id_label},
            "known predictions": {"box_data": known_pred_box_data, "class_labels": pred_class_id_label},
            "unknown predictions": {"box_data": unknown_pred_box_data, "class_labels": pred_class_id_label},
            "background predictions": {"box_data": background_pred_box_data, "class_labels": pred_class_id_label}}

    masks = None
    if "semantic_segmentation" in known_targets:
        ss_mask = known_targets["semantic_segmentation"].cpu().numpy()
        OBD_ss_mask = known_targets["semantic_segmentation_OBD"].cpu().numpy()
        masks = {"OBD_ground_truth": {"mask_data": OBD_ss_mask, "class_labels": {0: 'objects', 1: 'background', 2: 'drivable'}},
                 "ground_truth": {"mask_data": ss_mask, "class_labels": semantic_segmentation_class_id_label}}

    if display:

        print('img :', images)
        print("known :", known_predictions)
        print("unknown :", unknown_predictions)

        # Assuming img, known, and unknown are your variables
        img = images.cpu().numpy()  # Convert tensor to numpy array
        known_boxes = known_predictions['boxes'].cpu().numpy()
        known_labels = known_predictions['labels'].cpu().numpy()
        unknown_boxes = unknown_predictions['boxes'].cpu().numpy()

        # Plotting the image
        dpi = 300
        max_resolution = (img.shape[1] * dpi, img.shape[0] * dpi)
        fig, ax = plt.subplots(1, figsize=(12, 12))
        #fig, ax = plt.subplots(figsize=(max_resolution[0] / dpi, max_resolution[1] / dpi), dpi=dpi)
        ax.imshow(img.transpose(1, 2, 0))  # Transpose to (height, width, channels)

        # Plotting known bounding boxes in green
        for box, label in zip(known_boxes, known_labels):
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=3, edgecolor='dodgerblue', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1]-2, gt_class_id_label[label], color='dodgerblue', fontsize=12, ha='left', va='bottom')


        # Plotting unknown bounding boxes in blue
        for box in unknown_boxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

        ax.axis('off')
        plt.savefig('img_output/image_' + img_id + '.png', bbox_inches='tight', pad_inches=0)

    return wandb.Image(images, boxes=boxes, masks=masks)

