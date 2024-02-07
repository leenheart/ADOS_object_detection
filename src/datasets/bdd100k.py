import json
import orjson
import glob
import torch

from torchvision import transforms as T, datasets
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from plot_data import make_image_labels

class BDD100k_dataset(Dataset):

    def _get_json_targets(self, scale):

        #print("Rescale images and boxes at : ", scale)
        #self.labels, self.boxes, new_images_name = [], [], []
        self.labels, self.boxes, self.knowns, self.unknowns, new_images_name = [], [], [], [], []

        x_scale, y_scale = scale[1], scale[0]
        #x_scale, y_scale = 1, 1

        # Open the json file
        with open(self.root_directory + "/targets.json", 'r') as f:
            #targets = json.load(f)
            json_data = f.read()
            targets = orjson.loads(json_data)

        # Go through each targets
        for target in targets:

            # Check we got the image 
            if not target["name"].split('.')[0] in self.images_name:
                #print("we don't have the image corresponding to the target")
                continue

            #label, boxe = [], []
            label, boxe, known, unknown = [], [], [], []
            # Go trougth each labels
            for obj in target["labels"]:

                if not obj["category"] in self.classes_names:
                    continue

                box2d = obj["box2d"]
                boxe.append([box2d["x1"] * x_scale, box2d["y1"] * y_scale, box2d["x2"] * x_scale, box2d["y2"] * y_scale])
                label_name = obj["category"]
                if label_name == "cyclist":
                    label_name = "person"

                label.append(np.where(self.classes_names == label_name)[0][0])
                known.append(label_name in self.known_classes)
                unknown.append(label_name in self.unknown_classes)

            if label == [] or targets == [] :
                #TODO handle the case of images without labels
                continue

            self.labels.append(label)
            self.boxes.append(boxe)
            self.knowns.append(known)
            self.unknowns.append(unknown)
            boxe, label, known, unknown = [], [], [], []

            # Save the new order of images
            new_images_name.append(target["name"].split('.')[0])

            if len(new_images_name) >= self.max_size:
                break

        self.images_name = new_images_name
                

    def __init__(self, dataset_cfg, root_directory, max_size, classes_as_known, classes_as_unknown, combine=False, transform=None):
        super().__init__()


        self.known_classes = classes_as_known
        self.unknown_classes = classes_as_unknown

        semantics=dataset_cfg.semantics
        resize=(dataset_cfg.image_resize_height, dataset_cfg.image_resize_width)
        image_size=(dataset_cfg.image_height, dataset_cfg.image_width)

        #self.classes_names = ["background", "car", "bus", "truck", "motor", "bike", "pedestrian", "rider", "train"]
        if semantics and combine:
            self.classes_names = np.array(["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign", "road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"])
        else:
            self.classes_names = np.array(["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"])

        self.classes_names_seg = np.array(["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motor", "bike"])

        self.nb_classes = len(self.classes_names)

        self.root_directory = root_directory
        self.semantics = semantics
        self.combine = combine
        self.max_size = max_size #TODO

        #print("Init bdd100k dataset ", max_size, self.root_directory)

        # Load all images paths
        if self.semantics :
            self.images_path = glob.glob(self.root_directory + "/semantic_segmentation_images/*.png")
        else :
            self.images_path = glob.glob(self.root_directory + "/images/*.jpg")

        # Construct image name list
        self.images_name = []
        for i, image_path in enumerate(self.images_path):

            image_name = image_path.split("/")[-1].split(".")[0]
            self.images_name.append(image_name)

        # Load Datasets targets
        self._get_json_targets((resize[0]/image_size[0], resize[1]/image_size[1]))

        # Init transforms
        if transform != None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(size=resize),
                ])

        self.transform_seg = T.Compose([
            T.Resize(size=resize, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            ])


        self.classes_names_seg = np.array(["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motor", "bike"])


        #self.class_mapping_id = {}
        self.class_name_mapping_id = {}
        for i, name in enumerate(self.classes_names_seg):
            #self.class_mapping_id[i] = torch.tensor(value)
            self.class_name_mapping_id[i] = name 
        self.class_name_mapping_id[255] = "unknown"

        self.semantic_segmentation_classes_as_objects = [4, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 255]
        self.semantic_segmentation_classes_as_background = [2, 3, 5, 8, 10]
        self.semantic_segmentation_classes_as_drivable = [0, 1, 9]
        self.semantic_segmentation_classes_as_background_without_drivable = [i for i in self.semantic_segmentation_classes_as_background if i not in self.semantic_segmentation_classes_as_drivable]

        print("seg classes names mapping :", self.class_name_mapping_id)

    def __len__(self):

        return min(len(self.images_name), self.max_size)


    def semantic_segmentation_to_OBD(self, semantic_segmentation):

        semantic_segmentation_OBD = semantic_segmentation.detach().clone()

        mask_drivable = torch.isin(semantic_segmentation, torch.tensor(self.semantic_segmentation_classes_as_drivable))
        mask_objects = torch.isin(semantic_segmentation, torch.tensor(self.semantic_segmentation_classes_as_objects))
        mask_background = torch.isin(semantic_segmentation, torch.tensor(self.semantic_segmentation_classes_as_background_without_drivable))

        semantic_segmentation_OBD[mask_objects] = 0
        semantic_segmentation_OBD[mask_background] = 1
        semantic_segmentation_OBD[mask_drivable] = 2

        return semantic_segmentation_OBD

    def __getitem__(self, index):

        self.images_name[index]
        image = Image.open(self.root_directory + "/images/" + self.images_name[index] + ".jpg").convert("RGB")
        image = self.transform(image)

        targets = {"boxes": self.boxes[index], "labels": self.labels[index], "name": self.images_name[index], "knowns": self.knowns[index], "unknowns": self.unknowns[index], "image_id": self.images_name[index]}

        if self.semantics:
            semantic_segmentation_2d = (self.transform_seg(Image.open(self.root_directory + "/semantic_segmentation_masks/" + self.images_name[index] + ".png"))[0, :, :] * 255).long()
            targets["semantic_segmentation"] = semantic_segmentation_2d
            targets["semantic_segmentation_OBD"] = self.semantic_segmentation_to_OBD(semantic_segmentation_2d)
            #targets["semantics"] = self.transform_seg(Image.open(self.root_directory + "/semantic_segmentation_images/" + self.images_name[index] + ".png").convert("RGB"))

            """
            targets_seg_before = targets["seg_masks"].clone()
            # Change seg labels by combine labels
            if self.combine:
                for i, seg_class in enumerate(self.classes_names_seg):
                    targets["seg_masks"][targets_seg_before == i] = np.where(self.classes_names == seg_class)[0][0]
            #   , "semantic_segmentation": semantic_segmentation_2d, "semantic_segmentation_OBD": self.semantic_segmentation_to_OBD(semantic_segmentation_2d)
            """


        return image, targets

    def get_image_with_labels(self, index=0):

        image, targets = self.__getitem__(index)

        return make_image_labels(image, targets)
