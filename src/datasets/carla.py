import glob
import json
import torch
import numpy as np

import matplotlib.pyplot as plt

from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image

from plot_data import make_image_labels
from plot_data import make_image_labels_without_cpu

class Carla_dataset(Dataset):

    def _get_json_targets(self, scale):

        self.labels, self.boxes, self.knowns, self.unknowns, new_images_name = [], [], [], [], []

        #x_scale, y_scale = scale[1], scale[0]
        x_scale, y_scale = 1, 1

        for image_name in self.images_name:

            try:
                f = open(self.root_directory + "/labels/" + image_name.split('.')[0] + ".json", "r")
            except FileNotFoundError:
                print("Warning ! Label not found for label :", image_name)
            else:
                with f:

                    targets = json.load(f)
                    label, boxe, known, unknown = [], [], [], []
                    for target in targets:
                        boxe.append([target["min_x"], target["min_y"], target["max_x"], target["max_y"]])

                        if target[self.json_label_name] == "cyclist":
                            label_name = "pedestrian"
                        else:
                            label_name = target[self.json_label_name] 

                        label.append(self.class_to_id[label_name])
                        known.append(label_name in self.known_classes)
                        unknown.append(label_name in self.unknown_classes)

                    self.boxes.append(boxe)
                    self.labels.append(label)
                    self.knowns.append(known)
                    self.unknowns.append(unknown)
                    boxe, label, known, unknown = [], [], [], []





    def __init__(self, dataset_cfg, root_directory, max_size, classes_as_known, classes_as_unknown, combine=None, transform=None):
        super().__init__()

        resize=(dataset_cfg.image_resize_height, dataset_cfg.image_resize_width)
        image_size=(dataset_cfg.image_height, dataset_cfg.image_width)

        self.max_size = max_size 
        self.known_classes = classes_as_known
        self.unknown_classes = classes_as_unknown
        self.root_directory = root_directory
        self.json_label_name = dataset_cfg.json_label_name

        self.classes_names = dataset_cfg.classes_names
        self.class_to_id = {}
        for i, classe in enumerate(self.classes_names):
            self.class_to_id[classe] = i

        print("Init Carla ", max_size)

        # Load all images paths
        self.images_path = glob.glob(self.root_directory + "/rgb/*.png")
        print("looking at data at : ",self.root_directory + "/rgb/*.png",  " find nb images : ", len(self.images_path)) 


        # Construct image name list
        self.images_name = []
        for i, image_path in enumerate(self.images_path):

            image_name = image_path.split("/")[-1].split(".")[0]
            self.images_name.append(image_name)

        # Load Datasets targets
        self._get_json_targets((resize[0]/image_size[0], resize[1]/image_size[1]))
        self.nb_classes = len(self.classes_names)
        print("Classes names : ", self.classes_names)
        print("classes_as_know : ", classes_as_known)

        # Init transforms
        if transform != None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(size=resize),
                T.ToTensor(),
                ])

        self.transform_seg = T.Compose([
            T.Resize(size=resize, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            ])
        class_mapping = {
        'unlabeled': (0, 0, 0),
        'Buildings': (70, 70, 70),
        'Fences': (100, 40, 40),
        'Other': (55, 90, 80),
        'Pedestrians': (220, 20, 60),
        'Rider': (255, 0, 0),
        'pole': (153, 153, 153),
        'RoadLines': (157, 234, 50),
        'Roads': (128, 64, 128),
        'Sidewalks': (244, 35, 232),
        'Vegetation': (107, 142, 35),
        'Bicycle' : (119,  11,  32),
        'Bus': (0, 60, 100),
        'Car': (0, 0, 142),
        'Truck': (0, 0, 70),
        'Motorcycle': (0, 0, 230),
        'vehicle': (0, 0, 142),
        'Walls': (102, 102, 156),
        'traffic_sign': (220, 220, 0),
        'Sky': (70, 130, 180),
        'Ground': (81, 0, 81),
        'Bridge': (150, 100, 100),
        'RailTrack': (230, 150, 140),
        'GuardRail': (180, 165, 180),
        'traffic_light': (250, 170, 30),
        'Static': (110, 190, 160),
        'Dynamic': (170, 120, 50),
        'Water': (45, 60, 150),
        'Terrain': (145, 170, 100)
        }

        self.class_mapping_id = {}
        self.class_name_mapping_id = {}
        i = 0
        for key, value in class_mapping.items():
            self.class_mapping_id[i] = torch.tensor(value)
            self.class_name_mapping_id[i] = key
            i += 1

        self.semantic_segmentation_classes_as_objects = [4, 5, 11, 12, 13, 14, 15, 16, 18, 24, 25, 26]
        self.semantic_segmentation_classes_as_background = [1, 2, 3, 7, 8, 9, 10, 17, 19, 20, 21, 22, 23, 27, 28, 6] # 6 is pole
        self.semantic_segmentation_classes_as_drivable = [0, 7, 8, 9, 20, 28] # 6 is pole
        self.semantic_segmentation_classes_as_background_without_drivable = [i for i in self.semantic_segmentation_classes_as_background if i not in self.semantic_segmentation_classes_as_drivable]

        print("seg classes names mapping :", self.class_name_mapping_id)

    def semantic_segmentation_to_OBD(self, semantic_segmentation):

        semantic_segmentation_OBD = semantic_segmentation.detach().clone()

        mask_drivable = torch.isin(semantic_segmentation, torch.tensor(self.semantic_segmentation_classes_as_drivable))
        mask_objects = torch.isin(semantic_segmentation, torch.tensor(self.semantic_segmentation_classes_as_objects))
        mask_background = torch.isin(semantic_segmentation, torch.tensor(self.semantic_segmentation_classes_as_background_without_drivable))

        semantic_segmentation_OBD[mask_objects] = 0
        semantic_segmentation_OBD[mask_background] = 1
        semantic_segmentation_OBD[mask_drivable] = 2

        
        return semantic_segmentation_OBD

    def __len__(self):

        return min(len(self.images_name), self.max_size)

    def __getitem__(self, index):

        self.images_name[index]
        image = Image.open(self.root_directory + "/rgb/" + self.images_name[index] + ".png").convert("RGB")
        image = self.transform(image)
        semantic_segmantation = Image.open(self.root_directory + "/semantic_segmentation/" + self.images_name[index] + ".png").convert("RGB")
        semantic_segmantation = self.transform(semantic_segmantation) * 255

        semantic_segmantation = semantic_segmantation.permute(1, 2, 0)
        semantic_segmentation_2d = torch.zeros((semantic_segmantation.shape[0], semantic_segmantation.shape[1]))

        for key, value in self.class_mapping_id.items():
            mask = torch.all(semantic_segmantation[..., :3] == value, dim=2)
            semantic_segmentation_2d[mask] = key

        targets = {"boxes": self.boxes[index], "labels": self.labels[index], "name": self.images_name[index], "knowns": self.knowns[index], "unknowns": self.unknowns[index], "image_id": self.images_name[index], "semantic_segmentation": semantic_segmentation_2d, "semantic_segmentation_OBD": self.semantic_segmentation_to_OBD(semantic_segmentation_2d)}

        return image, targets

    def get_image_with_labels(self, index=0):

        image, targets = self.__getitem__(index)

        return make_image_labels(image, targets)
