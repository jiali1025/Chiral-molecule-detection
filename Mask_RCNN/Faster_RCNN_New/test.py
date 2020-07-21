from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
import val_data
import train_data
import os
from detectron2.engine.defaults import DefaultPredictor

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file('C:/Users/yz18514/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')       # Cover the config from config file.
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.DEVICE = 'cuda:0'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.SOLVER.IMS_PER_BATCH = 4                # batch_size=4; iters_in_one_epoch = dataset_imgs/batch_size.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)
    predictor = DefaultPredictor(cfg) 
    trainer = DefaultTrainer(cfg)
    evaluator = COCOEvaluator("val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "val")
    inference_on_dataset(predictor.model, val_loader, evaluator)
# another equivalent way is to use trainer.test