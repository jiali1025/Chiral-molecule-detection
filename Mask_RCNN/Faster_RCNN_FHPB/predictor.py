
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import train_data
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
import matplotlib
import matplotlib.pyplot as plt


train_metadata = MetadataCatalog.get("train")
dataset_dicts = DatasetCatalog.get("train")

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(
        "C:/Users/yz18514/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 10000000000000000
    cfg.DATASETS.TRAIN = ("train", )
    predictor = DefaultPredictor(cfg)

    data_f = 'C:/Users/yz18514/detectron2/Faster_RCNN/test4.png'
    #Input_Image_Path_FHPB
    im = cv2.imread(data_f)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
               metadata=train_metadata,
               scale=1.5,
               instance_mode=ColorMode.IMAGE_BW
               )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    cv2.imshow("Predictor", img)
    cv2.waitKey(0)
    
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    instances = outputs["instances"]._fields

    for i in range(len(instances["scores"])):
        if instances["pred_classes"][i] == 0:
            color = 'r'
        elif instances["pred_classes"][i] == 1:
            color = 'b'
        box = instances["pred_boxes"][i].tensor[0]
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        r = min(box[2]-box[0], box[3]-box[1]) / 2*0.9
        circ = matplotlib.patches.Circle((x, y), r, linewidth=1, fill=False, color=color)
        ax.add_patch(circ)
        #plt.text(0.9, 0.9, "L", withdash=True)
        #plt.text(0.8, 0.9, "R")

        plt.savefig('C:/Users/yz18514/Desktop/00000005.png')