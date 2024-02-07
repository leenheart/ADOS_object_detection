import numpy as np
import torch
import random
import cv2 as cv
import copy
from torchvision.ops import box_area, box_iou
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from traditional_seg import calculate_edge_density, calculate_color_contrast, compute_lab_hist, calculate_standard_deviation, calculate_luminance

def postprocess_predictions(images, predictions, targets, scores_cfg, random=True, with_perfect_oro=False ):

    if with_perfect_oro and "semantic_segmentation_OBD" in targets[0]:
        add_object_on_drivable_score(predictions, targets, display=False)
        add_object_on_drivable_score(targets, targets, display=False)

    #add_boxes_area(predictions)
    return None, None
    return add_custom_scores(images, targets, predictions, scores_cfg, with_randoms=random)


def add_semantic_segmentation_pourcent_drivable(predictions, targets, semantic_segmentation_classes_as_drivable):

    if len(predictions) == 0:
        return

    for i, target in enumerate(targets):

        scores_ss = []

        for boxe in predictions[i]["boxes"]: 
            scores_ss.append(calculate_semantic_segmentation_pourcent(boxe, target["semantic_segmentation"], semantic_segmentation_classes_as_drivable, bigger_box_factor=0, display=False))

        predictions[i]["score_drivable_pourcent"] = torch.tensor(scores_ss, device=predictions[i]["boxes"].device)


def calculate_semantic_segmentation_pourcent(boxe, semantic_segmentation, semantic_segmentation_classes_as_objects, bigger_box_factor=0, display=False):

    x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())

    x_distance = x_max - x_min
    y_distance = y_max - y_min
    x_padding = max(1, int(x_distance * bigger_box_factor))
    y_padding = max(1, int(y_distance * bigger_box_factor))

    y_min = max(0 , y_min - y_padding)
    x_min = max(0 , x_min - x_padding)
    x_max = min(semantic_segmentation.shape[1], x_max + x_padding)
    y_max = min(semantic_segmentation.shape[0], y_max + y_padding)

    semantic_segmentation_boxe = semantic_segmentation[y_min:y_max, x_min:x_max]
    mask_pixel_is_object = torch.any(semantic_segmentation_boxe.cpu().unsqueeze(-1) == torch.tensor(semantic_segmentation_classes_as_objects), dim=-1)

    sum_pixel_as_object = mask_pixel_is_object.sum().item()
    ss_boxe_shape = semantic_segmentation_boxe.shape 
    sum_pixel = ss_boxe_shape[0] * ss_boxe_shape[1] 

    if sum_pixel != 0:
        if display:
            print("boxe :", x_min, y_min, x_max, y_max,  "sum pixel :", sum_pixel, " pixel objects :", sum_pixel_as_object, " score :", sum_pixel_as_object/sum_pixel)
            fig = plt.figure()
            plt.imshow(semantic_segmentation.cpu().numpy())
            fig = plt.figure()
            plt.imshow(semantic_segmentation_boxe.cpu().numpy())
            fig = plt.figure()
            plt.imshow(mask_pixel_is_object.cpu().numpy())
            plt.show()
        return sum_pixel_as_object/sum_pixel
    else:
        print("no pixel so 0")
        return 0

def add_semantic_segmentation_pourcent(predictions, targets, semantic_segmentation_classes_as_objects):

    if len(predictions) == 0:
        return

    for i, target in enumerate(targets):

        scores_ss = []

        for boxe in predictions[i]["boxes"]: 
            scores_ss.append(calculate_semantic_segmentation_pourcent(boxe, target["semantic_segmentation"], semantic_segmentation_classes_as_objects))

        predictions[i]["score_semantic_segmentation"] = torch.tensor(scores_ss, device=predictions[i]["boxes"].device)



def calculate_object_on_drivable_score(prediction, semantic_segmentation_OBD, pourcent_of_boxe=0.2, display=False, training=False) -> torch.tensor:

    # Use v2 of ORO (this is v1) !

    # 0 is Objet, 1 is Background and 2 is drivable
    scores = []

    if training:
        boxes = prediction
    else:
        boxes = prediction["boxes"]

    for i, boxe in enumerate(boxes): 

        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())

        width = max(2, pourcent_of_boxe * (x_max - x_min))
        height = max(2, pourcent_of_boxe * (y_max - y_min))

        low_x_min = max(0 , x_min - int(width/2))
        low_y_min = max(0 , y_max - int(height/2))
        low_x_max = min(semantic_segmentation_OBD.shape[1], x_max + int(width/2))
        low_y_max = min(semantic_segmentation_OBD.shape[0], y_max + int(height/2))

        semantic_segmentation_lower_boxe = semantic_segmentation_OBD[low_y_min:low_y_max, low_x_min:low_x_max]
        mask_drivable_lower_boxe = semantic_segmentation_lower_boxe == 2

        semantic_segmentation_inter_boxe = semantic_segmentation_OBD[low_y_min:y_max, x_min:x_max]
        mask_drivable_inter_boxe = semantic_segmentation_inter_boxe == 2
        mask_object_inter_boxe = semantic_segmentation_inter_boxe == 0

        sum_pixel_as_object = mask_object_inter_boxe.sum().item()
        sum_pixel_as_drivable= mask_drivable_lower_boxe.sum().item() - mask_drivable_inter_boxe.sum().item()

        ss_boxe_shape = semantic_segmentation_lower_boxe.shape 
        sum_pixel = ss_boxe_shape[0] * ss_boxe_shape[1] 

        scores.append((sum_pixel_as_object + sum_pixel_as_drivable) / sum_pixel)

        if sum_pixel != 0 and i%10 == 0:
            if display:
                print("boxe :", x_min, y_min, x_max, y_max, " score :", scores[-1])
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                # Plot 1
                axes[0, 0].imshow(semantic_segmentation_OBD.cpu().numpy())
                axes[0, 0].set_title('semantic_segmentation_OBD score : ' + str(scores[-1]))
                # Add a red rectangle on the first plot at the location of the second plot
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=4, edgecolor='red', facecolor='none')
                axes[0, 0].add_patch(rect)

                # Plot 2
                axes[0, 1].imshow(semantic_segmentation_OBD[y_min:y_max, x_min:x_max].cpu().numpy())
                axes[0, 1].set_title(f'semantic_segmentation_OBD[{y_min}:{y_max}, {x_min}:{x_max}]')

                # Plot 3
                axes[1, 0].imshow(semantic_segmentation_lower_boxe.cpu().numpy())
                axes[1, 0].set_title('semantic_segmentation_lower_boxe')

                # Plot 4
                axes[1, 1].imshow(semantic_segmentation_inter_boxe.cpu().numpy())
                axes[1, 1].set_title('semantic_segmentation_inter_boxe')

                plt.show()
                """
                fig = plt.figure()
                plt.imshow(semantic_segmentation_OBD.cpu().numpy())
                fig = plt.figure()
                plt.imshow(semantic_segmentation_OBD[y_min:y_max, x_min:x_max].cpu().numpy())
                fig = plt.figure()
                plt.imshow(semantic_segmentation_lower_boxe.cpu().numpy())
                fig = plt.figure()
                plt.imshow(semantic_segmentation_inter_boxe.cpu().numpy())
                plt.show()
                """

    return torch.tensor(scores, device=boxes.device)

def calculate_SS_object_pourcent(prediction, semantic_segmentation_OBD, pourcent_of_boxe=0.3, display=False, training=False) -> torch.tensor:

    scores = []

    if training:
        boxes = prediction
    else:
        boxes = prediction["boxes"]

    if display:
        fig2, ax = plt.subplots()
        ax.imshow(semantic_segmentation_OBD.cpu().numpy())

    for i, boxe in enumerate(boxes): 

        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(semantic_segmentation_OBD.shape[1], x_max)
        y_max = min(semantic_segmentation_OBD.shape[0], y_max)

        width = max(2, pourcent_of_boxe * (x_max - x_min))
        height = max(2, pourcent_of_boxe * (y_max - y_min))

        low_x_min = max(0 , x_min - int(width/2))
        low_y_min = max(0 , y_max - int(height/2))
        low_x_max = min(semantic_segmentation_OBD.shape[1], x_max + int(width/2))
        low_y_max = min(semantic_segmentation_OBD.shape[0], y_max + int(height/2))


        semantic_segmentation_boxe = semantic_segmentation_OBD[y_min:y_max, x_min:x_max]
        mask_object_in_boxe = semantic_segmentation_boxe == 0
        sum_pixel_as_object_in_box = mask_object_in_boxe.sum().item()
        ss_boxe_shape = semantic_segmentation_boxe.shape 
        sum_pixel_box = ss_boxe_shape[0] * ss_boxe_shape[1] 

        if sum_pixel_box == 0:
            scores.append(0)
        else:
            scores.append(sum_pixel_as_object_in_box/ sum_pixel_box)

        if display:
            if display:
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                # Plot 1
                axes[0, 0].imshow(semantic_segmentation_OBD.cpu().numpy())
                axes[0, 0].set_title('semantic_segmentation_OBD score : ' + str(scores[-1]))
                # Add a red rectangle on the first plot at the location of the second plot

                # Plot 2
                axes[0, 1].imshow(semantic_segmentation_OBD[y_min:y_max, x_min:x_max].cpu().numpy())
                axes[0, 1].set_title(f'semantic_segmentation_OBD[{y_min}:{y_max}, {x_min}:{x_max}]')

                # Plot 3
                axes[1, 0].imshow(semantic_segmentation_lower_boxe.cpu().numpy())
                axes[1, 0].set_title('semantic_segmentation_lower_boxe')

                # Plot 4
                axes[1, 1].imshow(semantic_segmentation_inter_boxe.cpu().numpy())
                axes[1, 1].set_title('semantic_segmentation_inter_boxe')
                plt.show()

            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((low_x_min, low_y_min), low_x_max - low_x_min, low_y_max - low_y_min, linewidth=1, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
    if display:
        plt.show()

    return torch.tensor(scores, device=boxes.device)

def calculate_object_on_drivable_score_v2(prediction, semantic_segmentation_OBD, pourcent_of_boxe=0.3, display=False, training=False) -> torch.tensor:


    # 0 is Objet, 1 is Background and 2 is drivable
    scores = []

    if training:
        boxes = prediction
    else:
        boxes = prediction["boxes"]

    if display:
        fig2, ax = plt.subplots()
        ax.imshow(semantic_segmentation_OBD.cpu().numpy())

    for i, boxe in enumerate(boxes): 

        x_min, y_min, x_max, y_max = int(boxe[0].item()), int(boxe[1].item()), int(boxe[2].item()), int(boxe[3].item())
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(semantic_segmentation_OBD.shape[1], x_max)
        y_max = min(semantic_segmentation_OBD.shape[0], y_max)

        width = max(2, pourcent_of_boxe * (x_max - x_min))
        height = max(2, pourcent_of_boxe * (y_max - y_min))

        low_x_min = max(0 , x_min - int(width/2))
        low_y_min = max(0 , y_max - int(height/2))
        low_x_max = min(semantic_segmentation_OBD.shape[1], x_max + int(width/2))
        low_y_max = min(semantic_segmentation_OBD.shape[0], y_max + int(height/2))


        semantic_segmentation_boxe = semantic_segmentation_OBD[y_min:y_max, x_min:x_max]
        mask_object_in_boxe = semantic_segmentation_boxe == 0
        sum_pixel_as_object_in_box = mask_object_in_boxe.sum().item()
        ss_boxe_shape = semantic_segmentation_boxe.shape 
        sum_pixel_box = ss_boxe_shape[0] * ss_boxe_shape[1] 


        semantic_segmentation_lower_boxe = semantic_segmentation_OBD[low_y_min:low_y_max, low_x_min:low_x_max]
        mask_drivable_lower_boxe = semantic_segmentation_lower_boxe == 2
        ss_boxe_shape = semantic_segmentation_lower_boxe.shape 
        sum_pixel_lower_boxe = ss_boxe_shape[0] * ss_boxe_shape[1] 

        semantic_segmentation_inter_boxe = semantic_segmentation_OBD[low_y_min:y_max, x_min:x_max]
        mask_drivable_inter_boxe = semantic_segmentation_inter_boxe == 2
        ss_boxe_shape = semantic_segmentation_inter_boxe.shape 
        sum_pixel_inter_boxe = ss_boxe_shape[0] * ss_boxe_shape[1] 

        sum_pixel_as_drivable_in_ext = mask_drivable_lower_boxe.sum().item() - mask_drivable_inter_boxe.sum().item()
        sum_pixel_ext = sum_pixel_lower_boxe - sum_pixel_inter_boxe

        if sum_pixel_box == 0 or sum_pixel_ext == 0:
            scores.append(0)
        else:
            scores.append(sum_pixel_as_drivable_in_ext / sum_pixel_ext)

        if display:
            if display:
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                # Plot 1
                axes[0, 0].imshow(semantic_segmentation_OBD.cpu().numpy())
                axes[0, 0].set_title('semantic_segmentation_OBD score : ' + str(scores[-1]))
                # Add a red rectangle on the first plot at the location of the second plot

                # Plot 2
                axes[0, 1].imshow(semantic_segmentation_OBD[y_min:y_max, x_min:x_max].cpu().numpy())
                axes[0, 1].set_title(f'semantic_segmentation_OBD[{y_min}:{y_max}, {x_min}:{x_max}]')

                # Plot 3
                axes[1, 0].imshow(semantic_segmentation_lower_boxe.cpu().numpy())
                axes[1, 0].set_title('semantic_segmentation_lower_boxe')

                # Plot 4
                axes[1, 1].imshow(semantic_segmentation_inter_boxe.cpu().numpy())
                axes[1, 1].set_title('semantic_segmentation_inter_boxe')
                plt.show()

            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((low_x_min, low_y_min), low_x_max - low_x_min, low_y_max - low_y_min, linewidth=1, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
    if display:
        plt.show()

    return torch.tensor(scores, device=boxes.device)


def set_tags(predictions, targets, batch_index, iou_threshold, considering_classes=True, tags_name="tags"):

    predictions[batch_index][tags_name] = torch.zeros(len(predictions[batch_index]["boxes"]), dtype=torch.bool, device=targets[batch_index]["boxes"].device)
    predictions[batch_index]["IoU_" + tags_name] = torch.zeros(len(predictions[batch_index]["boxes"]), device=targets[batch_index]["boxes"].device)
    targets[batch_index][tags_name] = torch.zeros(len(targets[batch_index]["boxes"]), dtype=torch.bool, device=targets[batch_index]["boxes"].device)

    if len(targets[batch_index]["labels"]) == 0:
        return

    # Calculate all iou between mask pred and mask gt boxes
    iou_all = box_iou(predictions[batch_index]["boxes"], targets[batch_index]["boxes"])

    # For each prediction
    for prediction_index, pred_label in enumerate(predictions[batch_index]["labels"]):

        # Set not corresponding classes IoU at zero if considering classes 
        if considering_classes:
            class_mask = targets[batch_index]["labels"] == pred_label
            iou_all[prediction_index] *= class_mask.float()

        # Select best match 
        best_match_index = torch.argmax(iou_all[prediction_index]).detach()

        # Good detection
        if iou_all[prediction_index][best_match_index] >= iou_threshold:

            # set tags to true positive
            predictions[batch_index][tags_name][prediction_index] = True
            predictions[batch_index]["IoU_" + tags_name][prediction_index] = iou_all[prediction_index][best_match_index]

            if targets[batch_index][tags_name][best_match_index] == False:
                # set gt box at matched
                targets[batch_index][tags_name][best_match_index] = True


def seperate_predictions_with_threshold_score(predictions, score_threshold):

    # Select only known
    good_predictions = []
    bad_predictions = []
    for prediction in predictions:
        good_prediction = {}
        bad_prediction = {}

        #filter out only those predicted labels that are recognized and considered valid based on the known classes you have defined.
        good_mask = prediction["scores"] >= score_threshold

        for key, value in prediction.items():

            if key == "custom_scores":
                good_new_custom_scores = {}
                bad_new_custom_scores = {}
                for key2, value2 in prediction["custom_scores"].items():
                    good_new_custom_scores[key2] = value2[good_mask.cpu()]
                    bad_new_custom_scores[key2] = value2[~good_mask.cpu()]
                good_prediction[key] = good_new_custom_scores
                bad_prediction[key] = bad_new_custom_scores
                continue

            if not torch.is_tensor(value):
                continue

            good_prediction[key] = value[good_mask]
            bad_prediction[key] = value[~good_mask]

        good_predictions.append(good_prediction)
        bad_predictions.append(bad_prediction)

    return good_predictions, bad_predictions


def seperate_predictions_into_known_and_unknown(predictions, known_classes, score_threshold):

    # Select only known
    known_predictions = []
    unknown_predictions = []
    for prediction in predictions:
        known_prediction = {}
        unknown_prediction = {}

        #filter out only those predicted labels that are recognized and considered valid based on the known classes you have defined.
        known_mask = torch.tensor([label in known_classes for label in prediction["labels"]], dtype=torch.bool)
        good_mask = prediction["scores"] >= score_threshold
        mask = known_mask & good_mask.cpu()

        for key, value in prediction.items():

            if key == "custom_scores":
                known_new_custom_scores = {}
                unknown_new_custom_scores = {}
                for key2, value2 in prediction["custom_scores"].items():
                    known_new_custom_scores[key2] = value2[mask.cpu()]
                    unknown_new_custom_scores[key2] = value2[~mask.cpu()]
                known_prediction[key] = known_new_custom_scores
                unknown_prediction[key] = unknown_new_custom_scores
                continue

            if not torch.is_tensor(value):
                continue

            known_prediction[key] = value[mask]
            unknown_prediction[key] = value[~mask]

        known_predictions.append(known_prediction)
        unknown_predictions.append(unknown_prediction)

    return known_predictions, unknown_predictions


def get_only_targets(targets, on_background=False, on_known=False, on_unknown=False):

    # Select only known
    filtered_targets = []
    for target in targets:
        filtered_target = {}
        for key, value in target.items():

            if key == "custom_scores":
                new_custom_scores = {}
                for key_cs, value_cs in value.items():
                    if not torch.is_tensor(value_cs):
                        continue

                    elif on_known:
                        new_custom_scores[key_cs] = value_cs[target['knowns']]

                    elif on_unknown:
                        new_custom_scores[key_cs] = value_cs[target['unknowns']]

                    elif on_background:
                        new_custom_scores[key_cs] = value_cs[torch.logical_not(torch.logical_or(target["knowns"], target["unknowns"]))]

                    else:
                        raise ValueError("Need on type to filter on !")

                filtered_target[key] = new_custom_scores
                continue

            if not torch.is_tensor(value):
                continue

            if value.numel() == 0 or key == "semantic_segmentation" or key == "semantic_segmentation_OBD":
                filtered_target[key] = value

            elif on_known:
                filtered_target[key] = value[target['knowns']]

            elif on_unknown:
                filtered_target[key] = value[target['unknowns']]

            elif on_background:
                filtered_target[key] = value[torch.logical_not(torch.logical_or(target["knowns"], target["unknowns"]))]

            else:
                raise ValueError("Need on type to filter on !")

        filtered_targets.append(filtered_target)

    return filtered_targets

def get_only_known_targets(targets):
    return get_only_targets(targets, on_known=True)

def get_only_unknown_targets(targets):
    return get_only_targets(targets, on_unknown=True)
        
def get_only_background_targets(targets):
    return get_only_targets(targets, on_background=True)

def add_boxes_area(predictions):

    for prediction in predictions:
       prediction["boxes_area"] = (prediction["boxes"][:, 2].int() - prediction["boxes"][:, 0].int()) * (prediction["boxes"][:, 3].int() - prediction["boxes"][:, 1].int())



def create_randoms(images, targets, score_cfg):
    randoms = []
    true_randoms = []

    for i, image in enumerate(images):

        if len(images) > 1:
            if i < len(images) - 1:
                boxes = copy.deepcopy(targets[i + 1]["boxes"])
            else:
                boxes = copy.deepcopy(targets[0]["boxes"])

            if "semantic_segmentation" in targets[0]:
                background_boxes_indexes = []
                for j, boxe in enumerate(boxes):
                    if calculate_semantic_segmentation_pourcent(boxe, targets[i]["semantic_segmentation"], score_cfg.semantic_segmentation_classes_as_objects) <= 0.05:
                        background_boxes_indexes.append(j)
                
                randoms.append({"boxes": boxes[background_boxes_indexes]})
            else:
                randoms.append({"boxes": boxes})

        boxes = []
        for j in range(len(targets[i]["boxes"])):
            x = np.random.randint(images[i].shape[2] - 4)
            y = np.random.randint(images[i].shape[1] - 4)
            boxes.append([x, y, x + np.random.randint(4, images[i].shape[2] - x), y + np.random.randint(4, images[i].shape[1] - y)])
        true_randoms.append({"boxes": torch.tensor(boxes)})


    return randoms, true_randoms

def add_object_on_drivable_score(predictions, targets, display=False):

    for i, target in enumerate(targets):

        score_targets = calculate_object_on_drivable_score_v2(targets[i], target["semantic_segmentation_OBD"], display=display)
        score_predictions = calculate_object_on_drivable_score_v2(predictions[i], target["semantic_segmentation_OBD"], display=display)

        targets[i]["oro"] = score_targets
        predictions[i]["oro"] = score_predictions

        score_targets = calculate_SS_object_pourcent(targets[i], target["semantic_segmentation_OBD"], display=display)
        score_predictions = calculate_SS_object_pourcent(predictions[i], target["semantic_segmentation_OBD"], display=display)

        targets[i]["sso"] = score_targets
        predictions[i]["sso"] = score_predictions

def add_custom_scores(images, targets, predictions, score_cfg, display=False, with_randoms=True):

    ring_factor = score_cfg.color_contrast_ring_factor
    inner_ring_factor = score_cfg.edge_density_inner_ring_factor

    if with_randoms:
        randoms, true_randoms = create_randoms(images, targets, score_cfg)

        if "semantic_segmentation" in targets[0]:
            add_semantic_segmentation_pourcent(randoms, targets, score_cfg.semantic_segmentation_classes_as_objects)
            
    else:
        randoms, true_randoms = None, None

    random_CC, true_random_CC = add_color_contrast_scores(images, targets, predictions, randoms, true_randoms, ring_factor, display=display)
    random_ED, true_random_ED = add_edge_density_scores(images, targets, predictions, randoms, true_randoms, inner_ring_factor, display=display)
    random_std, true_random_std = add_edge_standard_deviation(images, targets, predictions, randoms, true_randoms)
    random_luminance, true_random_luminance = add_luminance_contrast(images, targets, predictions, randoms, true_randoms)

    if "semantic_segmentation" in targets[0]:
        add_semantic_segmentation_pourcent(targets, targets, score_cfg.semantic_segmentation_classes_as_objects)
        add_semantic_segmentation_pourcent(predictions, targets, score_cfg.semantic_segmentation_classes_as_objects)
        add_semantic_segmentation_pourcent_drivable(targets, targets, score_cfg.semantic_segmentation_classes_as_drivable)
        add_semantic_segmentation_pourcent_drivable(predictions, targets, score_cfg.semantic_segmentation_classes_as_drivable)

    if "semantic_segmentation_OBD" in targets[0]:
        add_object_on_drivable_score(predictions, targets, display=display)

    if with_randoms:
        random_scores = {"boxes": randoms, "color_contrast_scores": random_CC, "edge_density_scores": random_ED, "std": random_std, "luminance": random_luminance}
        true_random_scores = {"boxes": true_randoms, "color_contrast_scores": true_random_CC, "edge_density_scores": true_random_ED, "std": true_random_std, "luminance": true_random_luminance}

        return random_scores, true_random_scores
    return None, None


def add_luminance_contrast(images, targets, predictions, randoms, true_randoms):

    customs_scores_concatenate_random = []
    customs_scores_concatenate_true_random = []
    for i, image in enumerate(images):

        img = np.transpose(image.cpu().numpy(), (1, 2, 0))
        img = (img*255).astype(np.uint8)

        luminance_targets = calculate_luminance(targets[i], img)
        luminance_predictions = calculate_luminance(predictions[i], img)

        targets[i]["custom_scores"]["luminance"] = torch.tensor(luminance_targets, device=targets[i]["boxes"].device)
        predictions[i]["custom_scores"]["luminance"] = torch.tensor(luminance_predictions, device=predictions[i]["boxes"].device)


        if randoms != None and true_randoms != None:
            if len(images) > 1:
                luminance_randoms = calculate_luminance(randoms[i], img)
            luminance_true_randoms = calculate_luminance(true_randoms[i], img)
        
            if len(images) > 1:
                customs_scores_concatenate_random.append(luminance_randoms)
            customs_scores_concatenate_true_random.append(luminance_true_randoms)

    return customs_scores_concatenate_random, customs_scores_concatenate_true_random

def add_edge_standard_deviation(images, targets, predictions, randoms, true_randoms):

    customs_scores_concatenate_random = []
    customs_scores_concatenate_true_random = []
    for i, image in enumerate(images):

        img = np.transpose(image.cpu().numpy(), (1, 2, 0))
        img = (img*255).astype(np.uint8)

        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        std_targets = calculate_standard_deviation(targets[i], img, gray)
        std_predictions = calculate_standard_deviation(predictions[i], img, gray)

        targets[i]["custom_scores"]["std"] = torch.tensor(std_targets, device=targets[i]["boxes"].device)
        predictions[i]["custom_scores"]["std"] = torch.tensor(std_predictions, device=predictions[i]["boxes"].device)


        if randoms != None and true_randoms != None:
            if len(images) > 1:
                std_randoms = calculate_standard_deviation(randoms[i], img, gray)
            std_true_randoms = calculate_standard_deviation(true_randoms[i], img, gray)
        
            if len(images) > 1:
                customs_scores_concatenate_random.append(std_randoms)
            customs_scores_concatenate_true_random.append(std_true_randoms)

    return customs_scores_concatenate_random, customs_scores_concatenate_true_random

def add_edge_density_scores(images, targets, predictions, randoms, true_randoms, inner_factor, display=False):

    customs_scores_concatenate_random = []
    customs_scores_concatenate_true_random = []
    for i, image in enumerate(images):

        img = np.transpose(image.cpu().numpy(), (1, 2, 0))
        img = (img*255).astype(np.uint8)
        edge_img = cv.Canny(img, 100, 200)

        ED_targets = calculate_edge_density(targets[i], img, edge_img, inner_factor, display=display)
        ED_predictions = calculate_edge_density(predictions[i], img, edge_img, inner_factor, display=display)

        targets[i]["custom_scores"]["edge_density"] = torch.tensor(ED_targets, device=targets[i]["boxes"].device)
        predictions[i]["custom_scores"]["edge_density"] = torch.tensor(ED_predictions, device=predictions[i]["boxes"].device)

        if randoms != None and true_randoms != None:
            if len(images) > 1:
                ED_randoms = calculate_edge_density(randoms[i], img, edge_img, inner_factor, display=display)
            ED_true_randoms = calculate_edge_density(true_randoms[i], img, edge_img, inner_factor, display=display)
        
            if len(images) > 1:
                customs_scores_concatenate_random.append(ED_randoms)
            customs_scores_concatenate_true_random.append(ED_true_randoms)

    return customs_scores_concatenate_random, customs_scores_concatenate_true_random


def add_color_contrast_scores(images, targets, predictions, randoms, true_randoms, ring_factor, display=False):

    customs_scores_concatenate_random = []
    customs_scores_concatenate_true_random = []
    for i, image in enumerate(images):

        img = np.transpose(image.cpu().numpy(), (1, 2, 0))
        hist_img = compute_lab_hist(img)

        color_contrast_targets = calculate_color_contrast(targets[i], img, hist_img, ring_factor, display=display)
        color_contrast_predictions = calculate_color_contrast(predictions[i], img, hist_img, ring_factor, display=display)
        targets[i]["custom_scores"] = {"color_contrast": torch.tensor(color_contrast_targets, device=targets[i]["boxes"].device)}
        predictions[i]["custom_scores"] = {"color_contrast": torch.tensor(color_contrast_predictions, device=predictions[i]["boxes"].device)}

        if randoms != None and true_randoms != None:
            if len(images) > 1:
                color_contrast_randoms = calculate_color_contrast(randoms[i], img, hist_img, ring_factor, display=display)
            color_contrast_true_randoms = calculate_color_contrast(true_randoms[i], img, hist_img, ring_factor, display=display)
        
            if len(images) > 1:
                customs_scores_concatenate_random.append(color_contrast_randoms)
            customs_scores_concatenate_true_random.append(color_contrast_true_randoms)

    return customs_scores_concatenate_random, customs_scores_concatenate_true_random
