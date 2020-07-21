from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

import os
import cv2
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
import torch
from detectron2.modeling import GeneralizedRCNNWithTTA
import pycocotools
import train_data
import val_data



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """

    cfg = get_cfg()                             # Copy default config transcript.
    args.config_file = 'C:/Users/yz18514/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    cfg.merge_from_file(args.config_file)       # Cover the config from config file.
    cfg.merge_from_list(args.opts)              # Cover the config from CLI parameters.
    

    cfg.CUDNN_BENCHMARK = True
    cfg.MODEL.DEVICE = 'cuda:0'

    # Change configuration parameters.
    cfg.DATASETS.TRAIN = ("train",)             # Name of trainning set.
    cfg.DATASETS.TEST = ("val",)                # Name of testing set.
    cfg.DATALOADER.NUM_WORKERS = 10

    
    cfg.INPUT.MAX_SIZE_TRAIN = 800              # maximum size of input image in training set.
    cfg.INPUT.MAX_SIZE_TEST = 800               # maximum size of input image in testing set.
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 580)       # minimum size of input image in training set.
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
    cfg.INPUT.MIN_SIZE_TEST = 480               # minimum size of input image in testing set.

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]

    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MASK_FORMAT = "polygon"

    # Pre-trained model weights
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  
    
    cfg.SOLVER.IMS_PER_BATCH = 1                # batch_size=4; iters_in_one_epoch = dataset_imgs/batch_size.
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512) 
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # According to the total number of training data and batch_size.
    # Calculate the number of iterations required for each epoch.
    ITERS_IN_ONE_EPOCH = int(3000 / cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE)

    # Specify the maximum number of iterations.
    cfg.SOLVER.MAX_ITER = 10000
    # Initial learning rate.
    cfg.SOLVER.BASE_LR = 0.001
    # Optimiser.
    cfg.SOLVER.MOMENTUM = 0.9
    # Weight decay.
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # Learning rate decay multiple.
    cfg.SOLVER.GAMMA = 0.01
    # The learning rate decays when iterate to a specified number of times.
    cfg.SOLVER.STEPS = (2000,)
    # Before training, do a warm-up exercise, the learning rate slowly increases the initial learning rate.
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # Number of warm-up iterations.
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"

    # Conduct an evaluation when iterate to a specified number of times.
    cfg.TEST.EVAL_PERIOD = 2500
    #cfg.MODEL.RPN.NMS_THRESH = 0.5

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 48000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 8000

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )