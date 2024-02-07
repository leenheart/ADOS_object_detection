import numpy as np
import pytorch_lightning as pl
import torch
import copy
import random
import json
import os
from multiprocessing import Pool
from collections import Counter

from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.datasets import Cityscapes, CocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2

from datasets.bdd100k import BDD100k_dataset
from datasets.coda import Coda_dataset
from datasets.carla import Carla_dataset


class DatasetWrapper(Dataset):

    def __init__(self, dataset, converter_classes_id, num_workers, predictions_path=None, with_predictions=False):
        super().__init__()

        self.dataset = dataset
        self.converter_classes_id = converter_classes_id

        self.with_predictions = with_predictions

        if self.dataset and with_predictions:

            print("Loading predictions")
            self.load_predictions(predictions_path, num_workers)


    def load_prediction(json_file_path):
        with open(json_file_path, "r") as json_file:
            loaded_data = json.load(json_file)
            predictions = loaded_data["predictions"]
            predictions["boxes"] = torch.tensor(predictions["boxes"])
            predictions["labels"] = torch.tensor(predictions["labels"])
            predictions["scores"] = torch.tensor(predictions["scores"])
            
            return predictions 

    def load_predictions(self, predictions_path, num_workers):
        self.predictions = []

        paths_file = predictions_path + "/" + str(len(self.dataset)) + "_predictions_paths.json"
        paths_file_exist = False
        if os.path.exists(paths_file):
            with open(paths_file, "r") as r_paths_file:
                loaded_data = json.load(r_paths_file)
                loaded_paths = loaded_data["predictions_paths"]
            print(len(loaded_paths), len(self.dataset))
            if len(loaded_paths) == len(self.dataset):
                paths_file_exist = True
                print("Paths file found")

        print("creating paths list")
        paths = []
        for i in range(len(self.dataset)):
            name = self.dataset[i][1]["image_id"]
            paths.append(predictions_path + "/" + str(name) + ".json")
            if paths_file_exist:
                if paths[i] != loaded_paths[i]:
                    print("PAths are differents !", paths[i], loaded_paths[i])
                    paths_file_exist = False
                elif i > 10:
                    break

        if paths_file_exist:
            print("using loaded paths")
            paths = loaded_paths
        else:
            print("Writing paths files")
            with open(paths_file, "w") as w_paths_file:
                data = {"predictions_paths": paths}
                json.dump(data, w_paths_file, indent=4)


        print("Loading json")
        with Pool(processes=num_workers) as pool:
            self.predictions = pool.map(DatasetWrapper.load_prediction, paths)

    def __len__(self):

        if self.dataset == None:
            return 0

        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        # Convert label id to correspond with de model prediction id
        if "labels" in item[1]:
            for i, label in enumerate(item[1]["labels"]):
                item[1]["labels"][i] = self.converter_classes_id[item[1]["labels"][i]]

        if self.with_predictions:
            item[1]["predictions"] = self.predictions[idx]

        return item


class DatasetsWrapper:

    def __init__(self, dataset_cfg, skip_train_load, model_classes_names, predictions_path, num_workers, combine_classes=False, transform=None):

        self.model_classes_names = model_classes_names
        self.dataset_classes_names = dataset_cfg.classes_names
        self.dataset_classes_background = []
        if "classes_background" in dataset_cfg:
            self.dataset_classes_background = dataset_cfg.classes_background

        self.semantic_segmentation_class_id_label = None

        # Create list of classes merged from pred and dataset
        self.merged_classes_names = copy.deepcopy(model_classes_names)
        for label in self.dataset_classes_names:
            if label in self.merged_classes_names:
                continue
            self.merged_classes_names.append(label)
            

        # Create converter of id classes from dataset to prediction
        self.dataset_converter_classes = []
        for classe in self.dataset_classes_names:
            self.dataset_converter_classes.append(self.merged_classes_names.index(classe))

        # Create a list of all known classes (id and names) (intersection between prediction and dataset classes)

        self.classes_as_known_id = []
        self.classes_as_known = []
        self.classes_as_unknown_id = []
        self.classes_as_unknown = []
        self.classes_as_background_id = []
        self.classes_as_background = []

        for i, classe in enumerate(self.dataset_classes_names):

            if classe in model_classes_names:
                self.classes_as_known_id.append(self.merged_classes_names.index(classe))
                self.classes_as_known.append(classe)

            elif classe in self.dataset_classes_background:
                self.classes_as_background_id.append(self.merged_classes_names.index(classe))
                self.classes_as_background.append(classe)

            else:
                self.classes_as_unknown_id.append(self.merged_classes_names.index(classe))
                self.classes_as_unknown.append(classe)

        # remove repetition
        self.classes_as_known_id = list(set(self.classes_as_known_id))
        self.classes_as_unknown_id = list(set(self.classes_as_unknown_id))
        self.classes_as_background_id = list(set(self.classes_as_background_id))
        self.classes_as_known = list(set(self.classes_as_known))
        self.classes_as_unknown = list(set(self.classes_as_unknown))
        self.classes_as_background = list(set(self.classes_as_background))

        self.classes_as_known_id.sort()
        self.classes_as_unknown_id.sort()
        self.classes_as_background_id.sort()

        # Check if any repetitive value 
        combined_list = self.classes_as_background_id + self.classes_as_known_id + self.classes_as_unknown_id
        counted_values = Counter(combined_list)

        print()
        print("Dataset classes names : ", self.dataset_classes_names)
        print("Model classes names : ", self.model_classes_names)
        print("Merged classes names : ", self.merged_classes_names)
        print()
        print("classes_as_know : ", self.classes_as_known, self.classes_as_known_id)
        print("classes_as_unknown : ", self.classes_as_unknown, self.classes_as_unknown_id)
        print("classes_as_background: ", self.classes_as_background, self.classes_as_background_id)
        print()

        if any(value > 1 for value in counted_values.values()):
            raise ValueError("Classes can't be in both type (unknown or known or background) ! values dict : ", counted_values)

        if dataset_cfg.name == "coda":
            print("Loading CODA dataset")
            self.dataset_train = None
            self.dataset_test = None
            self.dataset_val = Coda_dataset(dataset_cfg, dataset_cfg.path, dataset_cfg.max_size, self.classes_as_known, self.classes_as_unknown)

        elif dataset_cfg.name == "carla":
            print("Loading CARLA dataset")
            train_selected_indices = range(dataset_cfg.train_max_size)
            val_selected_indices = range(dataset_cfg.train_max_size, dataset_cfg.train_max_size + dataset_cfg.val_max_size)
            test_selected_indices = range(dataset_cfg.train_max_size + dataset_cfg.val_max_size, dataset_cfg.max_size)

            dataset = Carla_dataset(dataset_cfg, dataset_cfg.path, dataset_cfg.max_size, self.classes_as_known, self.classes_as_unknown)

            self.dataset_train = torch.utils.data.Subset(dataset, train_selected_indices)
            self.dataset_val = torch.utils.data.Subset(dataset, val_selected_indices)
            self.dataset_test = torch.utils.data.Subset(dataset, test_selected_indices)
            self.semantic_segmentation_class_id_label = dataset.class_name_mapping_id

        elif dataset_cfg.name == "bdd100k":
            print("Loading BDD100k dataset")
            self.dataset_test = BDD100k_dataset(dataset_cfg, dataset_cfg.test_path, dataset_cfg.test_max_size, self.classes_as_known, self.classes_as_unknown)
            self.dataset_val = BDD100k_dataset(dataset_cfg, dataset_cfg.val_path, dataset_cfg.val_max_size, self.classes_as_known, self.classes_as_unknown)

            # For val the semantics seg images does not have bbox targets in the targets.json file of val
            if not skip_train_load:
                self.dataset_train = BDD100k_dataset(dataset_cfg, dataset_cfg.train_path, dataset_cfg.train_max_size, self.classes_as_known, self.classes_as_unknown)
            else:
                self.dataset_train = None

            if self.dataset_train:
                self.semantic_segmentation_class_id_label = self.dataset_train.class_name_mapping_id
            else:
                self.semantic_segmentation_class_id_label = self.dataset_val.class_name_mapping_id

        elif dataset_cfg.name == "coco":
            print("Loading COCO dataset")

            if transform == None:
                if "image_resize_height" in dataset_cfg:
                    resize=(dataset_cfg.image_resize_height, dataset_cfg.image_resize_width)
                    transform = T.Compose([T.Resize(size=resize),T.ToTensor()])
                    transforms = T.Compose([T.Resize(size=resize)])
                else:
                    transforms = None
                    transform = T.Compose([T.ToTensor()])
            else :
                print("Transform", transform)


            self.dataset_train = CocoDetection(dataset_cfg.train_path + dataset_cfg.image_folder_name, dataset_cfg.train_path + dataset_cfg.annotation_file_name, transform=transform)#, target_transform=targets_transform)
            print(self.dataset_train)
            self.dataset_val = CocoDetection(dataset_cfg.val_path + dataset_cfg.image_folder_name, dataset_cfg.val_path + dataset_cfg.annotation_file_name, transform=transform)#, target_transform=targets_transform)

            self.dataset_train = wrap_dataset_for_transforms_v2(self.dataset_train)
            self.dataset_val = wrap_dataset_for_transforms_v2(self.dataset_val)

            #selected_indices = random.sample(range(len(self.dataset_val)), dataset_cfg.val_max_size)
            val_selected_indices = range(dataset_cfg.val_max_size)
            self.dataset_val = torch.utils.data.Subset(self.dataset_val, val_selected_indices)
            self.dataset_train = torch.utils.data.Subset(self.dataset_train, range(dataset_cfg.train_max_size))
            self.dataset_test = None


        else :
            raise Exception("Config Errors, no dataset name ", dataset_cfg.name, " can be hundle")


        # Add wrapper to each datasets
        self.dataset_train = DatasetWrapper(self.dataset_train, self.dataset_converter_classes, num_workers)
        self.dataset_val = DatasetWrapper(self.dataset_val, self.dataset_converter_classes, num_workers, with_predictions=dataset_cfg.with_predictions, predictions_path=predictions_path)
        self.dataset_test = DatasetWrapper(self.dataset_test, self.dataset_converter_classes, num_workers, with_predictions=dataset_cfg.with_predictions, predictions_path=predictions_path)



    def get_datasets(self):
        return (self.dataset_train, self.dataset_val, self.dataset_test)

    def get_classes_as_known(self):
        return self.classes_as_known_id 

    def get_classes_as_unknown(self):
        return self.classes_as_unknown_id 

    def get_classes_as_background(self):
        return self.classes_as_background_id 
    
