batch_size=4

# Evaluate for each dataset (CARLA, CODA and COCO), 3 models (Faster RCNN basic, Open Set Faster RCNN and ADOS faster rcnn) with two mode each (without prediction in background (background class are removed) and with unknown prediction from background class)
# All the outputs are append to the test_out.txt file and only the results output are display in terminal

# CARLA 
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco test_on_val=false model.keep_background=false dataloader.batch_size=$batch_size >> test_out.txt
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco model.load=True model.to_load=1-faster_rcnn_reset_rpn_iou_coco_final dataloader.batch_size=$batch_size model.ai_iou=true test_on_val=false model.keep_background=false >> test_out.txt
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=false model.unknown_detections_per_img=500 test_on_val=false dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 >> test_out.txt
#
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco test_on_val=false model.keep_background=true dataloader.batch_size=$batch_size >> test_out.txt
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco model.load=True model.to_load=1-faster_rcnn_reset_rpn_iou_coco_final model.ai_iou=true test_on_val=false model.keep_background=true dataloader.batch_size=$batch_size >> test_out.txt
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=true model.unknown_detections_per_img=500 test_on_val=false dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 >> test_out.txt
python src/main.py +experiments=test_carla_mess_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=true model.unknown_detections_per_img=500 test_on_val=false dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 model.perfect_oro=true >> test_out.txt

# CODA
python src/main.py +experiments=test_coda_faster_rcnn_pretrained_coco test_on_val=true model.keep_background=false dataloader.batch_size=$batch_size >> test_out.txt
python src/main.py +experiments=test_coda_faster_rcnn_pretrained_coco model.load=True model.to_load=1-faster_rcnn_reset_rpn_iou_coco_final dataloader.batch_size=$batch_size model.ai_iou=true test_on_val=true model.keep_background=false >> test_out.txt
python src/main.py +experiments=test_coda_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=false model.unknown_detections_per_img=500 test_on_val=true dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 >> test_out.txt

python src/main.py +experiments=test_coda_faster_rcnn_pretrained_coco test_on_val=true model.keep_background=true dataloader.batch_size=$batch_size >> test_out.txt
python src/main.py +experiments=test_coda_faster_rcnn_pretrained_coco model.load=True dataloader.batch_size=$batch_size model.to_load=1-faster_rcnn_reset_rpn_iou_coco_final model.ai_iou=true test_on_val=true model.keep_background=true >> test_out.txt
python src/main.py +experiments=test_coda_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=true model.unknown_detections_per_img=500 test_on_val=true dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 >> test_out.txt

## COCO
python src/main.py +experiments=test_coco_faster_rcnn_pretrained_coco test_on_val=true dataloader.batch_size=$batch_size model.keep_background=false >> test_out.txt
python src/main.py +experiments=test_coco_faster_rcnn_pretrained_coco model.load=True dataloader.batch_size=$batch_size model.to_load=1-faster_rcnn_reset_rpn_iou_coco_final model.ai_iou=true test_on_val=true model.keep_background=false >> test_out.txt
python src/main.py +experiments=test_coco_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=false model.unknown_detections_per_img=500 test_on_val=true dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 >> test_out.txt

python src/main.py +experiments=test_coco_faster_rcnn_pretrained_coco test_on_val=true dataloader.batch_size=$batch_size model.keep_background=true >> test_out.txt
python src/main.py +experiments=test_coco_faster_rcnn_pretrained_coco model.load=True dataloader.batch_size=$batch_size model.to_load=1-faster_rcnn_reset_rpn_iou_coco_final model.ai_iou=true test_on_val=true model.keep_background=true >> test_out.txt
python src/main.py +experiments=test_coco_faster_rcnn_pretrained_coco model.load=True model.to_load=6-train_faster_rcnn_rpn_iou_oro_on_carla_faster_rcnn_epoch_43 model.ai_oro=true model.keep_background=true model.unknown_detections_per_img=500 test_on_val=true dataloader.batch_size=$batch_size model.unknown_roi_head_oro_score_threshold=0.5 model.unknown_roi_head_iou_score_threshold=0.4 >> test_out.txt


