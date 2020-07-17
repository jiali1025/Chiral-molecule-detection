import os
from multiprocessing.spawn import freeze_support

from matplotlib import pyplot as plt
import numpy
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
import matplotlib
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode

if __name__ == '__main__':
    freeze_support()


    register_coco_instances("mol_val", {}, r"E:\PyProjects\Faster-RCNN\Report\testset1\img\val.json",
                            r"E:\PyProjects\Faster-RCNN\Report\testset1\img")
    mol_metadata = MetadataCatalog.get("mol_val")

    cfg = get_cfg()
    # cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TEST = ("mol_val",)
    cfg.DATASETS.TRAIN = ("mol_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.INPUT.MIN_SIZE_TEST = 120
    cfg.MODEL.PIXEL_MEAN = [127, 127, 127]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

    # cfg.MODEL.FPN.OUT_CHANNELS = 8
    # cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    # cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3"]
    # cfg.MODEL.RPN.IN_FEATURES = ["p2","p3"]
    cfg.MODEL.RPN.NMS_THRESH = 0.5
    cfg.MODEL.WEIGHTS = r"E:\PyProjects\Faster-RCNN\Results\Test\output\trained2.pth"
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 36000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000
    cfg.TEST.DETECTIONS_PER_IMAGE = 5000
    # cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    # cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 16
    # cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 1
    # cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 4
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 6

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")
    trainer = DefaultTrainer(cfg)
    trainer.test(cfg, predictor.model, evaluator)


    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "mol_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)


