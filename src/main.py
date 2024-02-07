import hydra
import torchvision
import os
import torch
import wandb
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig, OmegaConf

from datamodule import DataModule
from nn_models.model import Model
from log import Logger


def see(cfg: DictConfig):

    print("Look at :", cfg.dataset.name)
    raise Exception("See cmd not upd to date!")

    # Create Data Module
    data_module = DataModule(cfg.dataset, cfg.dataloader, classes_as_known=cfg.dataset.classes_names)

    data_module.display_val_batch(shuffle=False)
    data_module.display_test_batch()



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    torch.set_float32_matmul_precision('high')

    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
        print("Dataset config : ", cfg.dataset)

    skip_train_load = False
    use_custom_scores = False

    if cfg.cmd == "see":
        return see(cfg)

    elif cfg.cmd == "test":
        skip_train_load = not cfg.test_on_train
        cfg.dataloader.shuffle_in_training = False

    elif cfg.cmd == "scores":
        use_custom_scores = True

    elif cfg.cmd != "train":
        raise Exception("[ERROR] The cmd " + cfg.cmd + " is not hundle, we have train, test and see.")
        

    # Create Logger
    logger = None
    if cfg.logger.wandb_on:
        config = OmegaConf.to_container(cfg, resolve=True)
        logger = WandbLogger(name=cfg.cmd + "_" + cfg.model.name + "_on_" + cfg.dataset.name, project=cfg.logger.project_name, offline=cfg.logger.wandb_offline, config=config, job_type=cfg.cmd)

    #Get correct model transform for dataset
    transform = None
    if cfg.model.pretrained:
        if cfg.model.name == "faster_rcnn":
            transform = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

        elif cfg.model.name == "fcos":
            transform = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1.transforms()
            print("Using fcos pretrained transform")

        elif cfg.model.name == "retina_net":
            transform = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

        elif cfg.model.name == "yolov8":
            transform=None

        else:
            raise Exception("Model no hundle !")

    if cfg.dataset.name == "bdd100k" and cfg.model.ai_oro:
        print("\nForcing bdd100k to be with semantic segmentation !!!")
        cfg.dataset.semantics = True

    # Create Datasets
    data_module = DataModule(cfg.dataset,
                             cfg.dataloader,
                             cfg.model.classes_names_pred,
                             cfg.model.save_predictions_path + cfg.model.name,
                             combine_classes=cfg.combine_classes,
                             skip_train_load=skip_train_load,
                             collate_fn=cfg.model.collate_fn,
                             transform=transform)

    if cfg.debug:
        print("CLASSES NAMES PREDICTED BY MODEL :", cfg.model.classes_names_pred)
        print("classes as known :", data_module.datasets.get_classes_as_known())
        print("classes as unknown :", data_module.datasets.get_classes_as_unknown())
        print("classes as background :", data_module.datasets.get_classes_as_background())
        print("data_module.datasets.semantic_segmentation_class_id_label : ", data_module.datasets.semantic_segmentation_class_id_label)

    cfg.model.rl_model_save_path += str(cfg.dataset.name) + "_"

    # Create Model
    model = Model(cfg.model,
                  cfg.metrics,
                  data_module.datasets.get_classes_as_known(),
                  classes_names_pred=cfg.model.classes_names_pred,
                  classes_names_gt=cfg.dataset.classes_names,  #data_module.datasets.merged_classes_names,              #cfg.dataset.classes_names,
                  classes_names_merged=data_module.datasets.merged_classes_names,
                  batch_size=cfg.dataloader.batch_size,
                  show_image=cfg.show_image,
                  use_custom_scores=use_custom_scores,
                  semantic_segmentation_class_id_label=data_module.datasets.semantic_segmentation_class_id_label)

    print("\n Init done \n Starting operations :")

    # Create Trainer
    trainer = pl.Trainer(accelerator='gpu',
                         logger=logger,
                         auto_scale_batch_size=False,
                         max_epochs=cfg.max_epochs,
                         log_every_n_steps=cfg.logger.log_every_n_steps,
                         accumulate_grad_batches=cfg.accumulate_gradient_nb_batch,
                         precision=cfg.precision_float,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         #callbacks=[EarlyStopping(monitor="val_map_known", mode="max")],
                         num_sanity_val_steps=0
                         )
                         

    # Save config into save directory
    with open(model.save_path + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)


    if cfg.cmd == "train":
        print("\nStart training")
        print("trainning set size :", len(data_module.dataset_train), "\n")
        trainer.fit(model=model, datamodule=data_module)
        print("\nEnd training\n")


    # Test on all datasets depending of the config
    if cfg.test_on_train:
        print("Start testing on training dataset")
        trainer.test(model=model, dataloaders=data_module.train_dataloader())

    if cfg.test_on_val:
        print("Start testing on val dataset")
        trainer.test(model=model, dataloaders=data_module.val_dataloader())

    if cfg.test_on_test:
        print("Start testing on test dataset")
        trainer.test(model=model, dataloaders=data_module.test_dataloader())

    wandb.run.summary["Intersection histograms unknown FP and TP on Color Contrast"] = model.test_metrics.intersection_unknown_FP_TP_on_CC
    wandb.run.summary["Intersection histograms random and targets on Edge density"] = model.test_metrics.intersection_random_and_targets_on_ED

    wandb.finish()

    return model.test_metrics.intersection_random_and_targets_on_ED

if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system')
    torchvision.disable_beta_transforms_warning()
    main()
