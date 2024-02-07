# Addressing Open-set Object Detection for Autonomous Driving perception: A focus on road objects

Open Set Object Detection framework specialized for Autonomous Driving perception

## Abstract

Autonomous Vehicles (AVs) are expected to take safe and efficient decisions. Hence, AVs need to be robust to real world situations and especially to cope with open world setting i.e. the ability to handle novelty such as unseen objects. Classical object detection models are trained to recognize a predefined set of classes but struggle to generalize well to novel classes at inference stage. Open-Set Object Detection (OSOD) aims to address the challenge of correctly detecting objects from unknown classes. However, autonomous driving systems possess specific open-set characteristics that are not yet covered by OSOD methods. Indeed, a detection error could lead to catastrophic events, emphasizing the importance of prioritizing the quality of box detection over quantity. Also the specific characteristics of objects encountered in road scenes could be leveraged to improve their detection in the open-world setting. In this vein, we introduce a new definition of objects of interest for autonomous driving perception, enabling the proposition of an AV specialized open-set object detector coined ADOS. The proposed model uses a new score, learnt with the background ground truth of the semantic segmentation. This On Road Object score measures whether the object is on drivable areas, enhancing the selection of unknown detection. Experimental evaluations are conducted on simulated and real world datasets and reveal that our method outperforms the baseline approaches in unknown object detection settings with the same detection performance on known objects as the closed-set object detector.

## Installation

### Requirements

Works for :

- python: 3.10.12
- torch: 2.1.0
- torchvision: 0.16.0
- pytorch lightning: 1.9.5
- hydra: 1.3.2


We do not ensure compatibility with any other versions at this time.


### Datasets

For the COCO dataset, download the 2017 train and val annotations and images [here](https://cocodataset.org/#download).
For the BDD100k dataset, download the 100k images [here](https://doc.bdd100k.com/download.html).
For the Coda dataset, download the 2022 test and val dataset [here](https://coda-dataset.github.io/download.html).
For the Carla dataset, download [here](https://nuage.insa-rouen.fr/index.php/s/wWkLy8gB7SgwF2N?path=CornerSet_Object_level_06_11_2023).

The repository tree dataset structure is as follows:


├── bdd100k
|   ├── train
|   |   ├── images
|   |   |   └──\*.jpg
|   |   └── annotations.json
|   ├── val
|   |   ├── images
|   |   |   └──\*.jpg
|   |   └── annotations.json
|   └── test
|       ├── images
|       |   └──\*.jpg
|       └── annotations.json
|
├── coda
|   ├── val
|   |   ├── images
|   |   |   └──\*.jpg
|   |   └── annotations.json
|   └── test
|       ├── images
|       |   └──\*.jpg
|       └── annotations.json
|
├── coco
|   ├── val
|   |   ├── images
|   |   |   └──\*.jpg
|   |   └── targets.json
|   └── train
|       ├── images
|       |   └──\*.jpg
|       └── targets.json
|
├── carla
|   └── mess
|       ├── rgb
|       |   └── \*.png
|       └── labels
|           └── \*.json



## Configuration

The configuration parameters are defined in YAML format under the `config/` directory. Modify these files to make the code work on your machine.

The first config file to make is your PC config in `config/pc_config/`. It contains: The dataset's path, the path to where the models are saved, the number of cores your CPU has (`dataloader.num_worker`), and the batch size your machine can handle. `laptop_corentin.yaml` is an example. Make sure to have the correct path where all your datasets are located.

In the `experiments` folder, there are configs for all the experiment combinations.

## Usage

Some bookkeeping needs to be done for the code. We will update these shortly.

## Documentation

We base all our modifications of Faster R-CNN on the PyTorch implementation for reproducibility purposes.


## Contact Information

For any questions or inquiries, feel free to contact Corentin Bunel at <mailto:corentin.bunel@insa-rouen.fr>


## Acknowledgements:


This work is funded by the French National Research Agency as part of the MultiTrans project under reference ANR-21-CE23-0032.

See [MultiTrans](https://anr-multitrans.github.io/) for more information about the project.


## Citation

```
```
