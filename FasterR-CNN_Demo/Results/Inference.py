from multiprocessing.spawn import freeze_support
import numpy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib
import os

if __name__ == '__main__':
    freeze_support()

    im = cv2.imread(r"E:\PyProjects\FasterR-CNN_Demo\Database\testset1\Test.png")  # Image to be detected

    # Register trainsets
    trainset1_directory = r"E:\PyProjects\FasterR-CNN_Demo\Database\trainset1"  # Directory of dataset
    register_coco_instances("train1", {}, trainset1_directory + r"Annotations\0.json",
                            # Annotations of the dataset
                            trainset1_directory + r"\Images")  # Directory of images of the dataset

    meta = MetadataCatalog.get("train1").set(thing_classes=["L", "R"])  # Set class label

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"  # Use 'cpu' instead if cuda is not available
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

    cfg.INPUT.MAX_SIZE_TEST = 600  # Keep the size consistent with the range used during training

    cfg.MODEL.WEIGHTS = r"E:\PyProjects\FasterR-CNN_Demo\Results\output\model_final.pth"  # Well-trained model
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 24000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 4000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Only show results with high scores
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    predictor = DefaultPredictor(cfg)

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im2 = numpy.zeros_like(im)
    for i in range(3):
        im2[:, :, i] = gray  # Convert to grayscale for inference

    outputs = predictor(im2)
    v = Visualizer(im, metadata=meta, scale=1.2, instance_mode=2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Results", out.get_image())
    cv2.waitKey(0)


    instances = outputs["instances"]._fields
    matplotlib.pyplot.imshow(im2)
    ax = matplotlib.pyplot.gca()
    for i in range(len(instances["scores"])):
        if instances["pred_classes"][i] == 0:
            color = "deepskyblue"
        elif instances["pred_classes"][i] == 1:
            color = "orange"
        box = instances["pred_boxes"][i].tensor[0]
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        r = min(box[2] - box[0], box[3] - box[1]) / 2 * 0.9
        circ = matplotlib.patches.Circle((x, y), r, linewidth=1, fill=False, color=color)
        ax.add_patch(circ)

    matplotlib.pyplot.savefig(os.environ['USERPROFILE'] + '\Desktop' + r"\result.png")

