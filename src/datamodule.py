import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from plot_data import make_image_labels
from datasets.datasets_wrapper import DatasetsWrapper


"""
list of dictionary
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
"""

def coco_collate_fn(data):

    images = []
    targets = []

    for d in data:
        images.append(d[0])

        if len(d[1].keys()) <= 2:
            boxes = []
            labels = []
        else:
            boxes = d[1]["boxes"]
            labels = d[1]["labels"]

        if "predictions" in d[1]:
            predictions_boxes = d[1]["predictions"]["boxes"]
            predictions_labels = d[1]["predictions"]["labels"]
            predictions_scores = d[1]["predictions"]["scores"]
            targets.append({"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "predictions": {"boxes": predictions_boxes, "labels": predictions_labels, "scores": predictions_scores}})
        else:
            targets.append({"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)})

        targets[-1]["knowns"] = torch.ones_like(targets[-1]["labels"], dtype=torch.bool)
        targets[-1]["unknowns"] = torch.zeros_like(targets[-1]["labels"], dtype=torch.bool)
        targets[-1]["name"] = str(d[1]["image_id"])

    return (images, targets)

def list_coco_collate_fn(data):
    images, targets = coco_collate_fn(data)
    return [images, targets]

def tensor_coco_collate_fn(data):
    images, targets = coco_collate_fn(data)
    return [torch.stack(images, dim=0), targets]

def collate_fn(data):

    images = []
    targets = []

    for d in data:
        images.append(d[0])
        boxes = d[1]["boxes"]
        labels = d[1]["labels"]

        if "predictions" in d[1]:
            predictions_boxes = d[1]["predictions"]["boxes"]
            predictions_labels = d[1]["predictions"]["labels"]
            predictions_scores = d[1]["predictions"]["scores"]
            targets.append({"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "predictions": {"boxes": predictions_boxes, "labels": predictions_labels, "scores": predictions_scores}})
        else:
            targets.append({"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)})

        targets[-1]["name"] = d[1]["image_id"]

        if "knowns" in d[1]:
            targets[-1]["knowns"] = torch.tensor(d[1]["knowns"])
        else:
            raise ValueError("Should have a known tensor from dataset to explain if the target box is known or not")
            targets[-1]["knowns"] = torch.ones_like(targets[-1]["labels"], dtype=torch.bool)

        if "unknowns" in d[1]:
            targets[-1]["unknowns"] = torch.tensor(d[1]["unknowns"])
        else:
            raise ValueError("Should have a unknown tensor from dataset to explain if the target box is unknown or not")
            targets[-1]["unknowns"] = torch.ones_like(targets[-1]["labels"], dtype=torch.bool)


        if "semantic_segmentation" in d[1]:
            targets[-1]["semantic_segmentation"] = d[1]["semantic_segmentation"]
            targets[-1]["semantic_segmentation_OBD"] = d[1]["semantic_segmentation_OBD"]

    return (images, targets)

def list_collate_fn(data):
    images, targets = collate_fn(data)
    return [images, targets]

def tensor_collate_fn(data):
    images, targets = collate_fn(data)
    return [torch.stack(images, dim=0), targets]

class DataModule(pl.LightningDataModule):

    def __init__(self, dataset_cfg, dataloader_cfg, model_classes_names, predictions_path, open_set=False, combine_classes=False, skip_train_load=False, collate_fn="list", transform=None):
        super().__init__()

        if collate_fn == "list":
            if dataset_cfg.name == "coco":
                collate_fn = list_coco_collate_fn
            else:
                collate_fn = list_collate_fn

        elif collate_fn == "tensor":
            if dataset_cfg.name == "coco":
                collate_fn = tensor_coco_collate_fn
            else:
                collate_fn = tensor_collate_fn

        else:
            raise Exception("Config Errors, no collate fonction name ", collate_fn, " can be hundle")

        self.data_loaders_args = {'batch_size': dataloader_cfg.batch_size, 'num_workers': dataloader_cfg.num_workers, 'drop_last': True, "collate_fn": collate_fn}#, "pin_memory": True}
        self.shuffle_in_training = dataloader_cfg.shuffle_in_training

        # Create datasets
        self.datasets = DatasetsWrapper(dataset_cfg, skip_train_load, model_classes_names, predictions_path, dataloader_cfg.num_workers, combine_classes=combine_classes, transform=transform)
        self.dataset_train, self.dataset_val, self.dataset_test = self.datasets.get_datasets()

        self.print_info()

    def print_info(self):

        if self.dataset_train == None:
            print("There is no train dataset")
        else:
            print("Train dataset size : ", len(self.dataset_train))#, self.dataset_train.images_name[:5])

        if self.dataset_val == None:
            print("There is no validation dataset")
        else:
            print("Val dataset size : ", len(self.dataset_val))#, self.dataset_val.images_name[:5])

        if self.dataset_test == None:
            print("There is no test dataset")
        else:
            print("Test dataset size : ", len(self.dataset_test))#, self.dataset_test.images_name[:5])

    def train_dataloader(self):

        if self.dataset_train == None:
            return None

        return DataLoader(self.dataset_train, **self.data_loaders_args, shuffle=self.shuffle_in_training)

    def val_dataloader(self, shuffle=False):

        if self.dataset_val == None:
            return None

        return DataLoader(self.dataset_val, **self.data_loaders_args, shuffle=shuffle)

    def test_dataloader(self):

        if self.dataset_test == None:
            return None

        return DataLoader(self.dataset_test, **self.data_loaders_args, shuffle=False)


    def get_single_image_target(self, index):
        # Get a single image and its target label from the test dataset
        image, target = self.dataset_val[index]
        return image, target

    def get_batch_image_with_labels(batch, classes_names, one=False):

        labeled_images = []

        images, targets = batch

        # Change labels number to name
        for target in targets:
            labels = target["labels"]
            labels_names = []
            for label in labels:
                labels_names.append(classes_names[label.detach()])
            target["labels"] = labels_names

            if one:
                break
            

        for i, image in enumerate(images):

            image_with_label = make_image_labels(image, targets[i])

            if "semantics" in targets[i]:
                labeled_images.append([image_with_label, targets[i]["semantics"], image.cpu().permute(1, 2, 0)])
            else:
                labeled_images.append([image_with_label])

            if one:
                break

        return labeled_images

    def display_train_batch(self):

        dataloader = self.train_dataloader()
        if dataloader == None:
            print("Empty trainning set, can't display batches")
        else:
            self.display_batch(dataloader, self.dataset_train.classes_names)

    def display_val_batch(self, shuffle):

        dataloader = self.val_dataloader(shuffle=shuffle)
        if dataloader == None:
            print("Empty validation set, can't display batches")
        else:
            self.display_batch(dataloader, self.dataset_val.classes_names)

    def display_test_batch(self):

        dataloader = self.test_dataloader()
        if dataloader == None:
            print("Empty test set, can't display batches")
        else:
            self.display_batch(dataloader, self.dataset_test.classes_names)


    def display_batch(self, dataloader, classes_names):

        labeled_images = DataModule.get_batch_image_with_labels(next(iter(dataloader)), classes_names)

        for image in labeled_images:

            fig = plt.figure()

            nb_subplot = len(image)
            for i in range(nb_subplot):
                fig.add_subplot(1, nb_subplot, i + 1)
                plt.imshow(image[i])
                plt.axis('off')

            fig.subplots_adjust(wspace=0, hspace=0)
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close(fig)
