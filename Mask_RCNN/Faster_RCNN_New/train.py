import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import mol_data
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os
setup_logger()



if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file('C:/Users/yz18514/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    
    cfg.DATASETS.TRAIN = ("mol",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset


    cfg.DATALOADER.NUM_WORKERS = 10  # 单线程


    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 610 # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 610 # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (15, 550) # 训练图片输入的最小尺寸，可以吃定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 610
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'


    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 clases L & R
    
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 48

    ITERS_IN_ONE_EPOCH = int(3000 / cfg.SOLVER.IMS_PER_BATCH)


    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 100) - 1


    cfg.SOLVER.BASE_LR = 0.001

    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9

    
    #权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    #cfg.TEST.EVAL_PERIOD = 100

    cfg.MODEL.DEVICE = 'cuda:0'

    cfg.freeze()    
    
    
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)  # faster, and good enough for this toy dataset

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()