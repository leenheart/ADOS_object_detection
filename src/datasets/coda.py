import glob
import json
import torch

import matplotlib.pyplot as plt

from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image

from plot_data import make_image_labels
from plot_data import make_image_labels_without_cpu

class Coda_dataset(Dataset):

    def _get_json_targets(self, scale):

        self.labels, self.boxes, self.knowns, self.unknowns, new_images_name, self.corner_cases = [], [], [], [], [], []

        #x_scale, y_scale = scale[1], scale[0]
        x_scale, y_scale = 1, 1

        # Open the json file
        with open(self.root_directory + "/annotations.json", 'r') as f:
            targets = json.load(f)
            self.categories = targets["categories"]

            self.id_to_classe = [0] * 44 
            #self.classes_names = []
            for i, categorie in enumerate(self.categories):
                #self.classes_names.append(categorie["name"])
                self.id_to_classe[categorie["id"]] = i

            current_image_id = 1
            label, boxe, known, unknown, corner_case = [], [], [], [], []

            # Go through each targets
            for target in targets["annotations"]:

                if current_image_id >= len(targets["images"]):
                    break

                # Check we got the image 
                if not targets["images"][current_image_id]["file_name"].split('.')[0] in self.images_name:
                    print("We don't have the image ", targets["images"][current_image_id]["file_name"].split('.')[0])
                    continue

                if target["image_id"] != current_image_id:
                    self.boxes.append(boxe)
                    self.labels.append(label)
                    self.knowns.append(known)
                    self.unknowns.append(unknown)
                    self.corner_cases.append(corner_case)
                    label, boxe, known, unknown, corner_case = [], [], [], [], []

                    # Save the new order of images
                    new_images_name.append(targets["images"][current_image_id - 1]["file_name"].split('.')[0])
                    current_image_id = target["image_id"]

                box = target["bbox"]
                boxe.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                label.append(self.id_to_classe[target["category_id"]])
                known.append(self.classes_names[label[-1]] in self.known_classes)
                unknown.append(self.classes_names[label[-1]] in self.unknown_classes)
                corner_case.append(target["corner_case"])

                if len(new_images_name) >= self.max_size:
                    break

        self.images_name = new_images_name



    #def __init__(self, root_directory, transform=None, max_size=None, size=None, resize=None, classes_as_known=None):
    def __init__(self, dataset_cfg, root_directory, max_size, classes_as_known, classes_as_unknown, combine=False, transform=None):
        super().__init__()

        resize=(dataset_cfg.image_resize_height, dataset_cfg.image_resize_width)
        image_size=(dataset_cfg.image_height, dataset_cfg.image_width)

        self.root_directory = root_directory
        self.max_size = max_size 
        self.known_classes = classes_as_known
        self.unknown_classes = classes_as_unknown
        self.classes_names = dataset_cfg.classes_names # rewrite classes names with those in cfg

        print("Init coda ", max_size)

        # Load all images paths
        self.images_path = glob.glob(self.root_directory + "/images/*.jpg")

        # Construct image name list
        self.images_name = []
        for i, image_path in enumerate(self.images_path):

            image_name = image_path.split("/")[-1].split(".")[0]
            self.images_name.append(image_name)

        # Load Datasets targets
        self._get_json_targets((resize[0]/image_size[0], resize[1]/image_size[1]))
        self.nb_classes = len(self.classes_names)

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


    def __len__(self):

        return min(len(self.images_name), self.max_size)

    def __getitem__(self, index):

        self.images_name[index]
        image = Image.open(self.root_directory + "/images/" + self.images_name[index] + ".jpg").convert("RGB")
        image = self.transform(image)

        targets = {"boxes": self.boxes[index], "labels": self.labels[index], "image_id": self.images_name[index], "knowns": self.knowns[index], "unknowns": self.unknowns[index]}

        #print(self.root_directory + "/images/" + self.images_name[index] + ".jpg")
        #print(targets)

        return image, targets

    def get_image_with_labels(self, index=0):

        image, targets = self.__getitem__(index)

        return make_image_labels(image, targets)
