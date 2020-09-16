from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer


# Register trainsets
trainset1_directory= r"E:\PyProjects\FasterR-CNN_Demo\Database\trainset1"  # Directory of dataset
register_coco_instances("train1", {}, trainset1_directory + r"\Annotations\dataset.json",  # Annotations of the dataset
                        trainset1_directory + r"\Images") # Directory of images of the dataset

# register more dataset if available
# trainset2_directory= r"\Database\trainset2"
# register_coco_instances("train1", {}, trainset2_directory + r"Annotations\dataset.json",
#                         trainset2_directory + r"\Images")

cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("train1",)  # We registered only one train set here
cfg.DATASETS.TEST = ()  # We dis not registered any test set here
cfg.DATALOADER.NUM_WORKERS = 0

cfg.INPUT.MAX_SIZE_TRAIN = 600
cfg.INPUT.MIN_SIZE_TRAIN = (100, 600)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'  # Resizing images during training time

cfg.SOLVER.IMS_PER_BATCH = 2 # A batch size of 2 is large enough for this trainset to converge
cfg.SOLVER.STEPS = (1500,)  # Learning rate decay at 1500 iterations
cfg.SOLVER.MAX_ITER = 2000  # 2000 iterations are enough for this dataset
cfg.SOLVER.GAMMA = 0.1  # Learning rate decay
cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.001
cfg.SOLVER.CHECKPOINT_PERIOD = 2000  # Save the model every 2000 iterations
cfg.SOLVER.WARMUP_ITERS = 100

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Number of classes the model is requierd to classification, "L" and "R" chirality
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 24000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 4000
cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.2

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]  # Adjust anchor sizes for our task
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]  # Adjust anchor shapes for our task
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

cfg.TEST.EVAL_PERIOD = 0  # Since we do not have testset here

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
