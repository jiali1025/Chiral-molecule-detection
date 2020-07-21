# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = 'C:/Users/yz18514/detectron2/Faster_RCNN/data/train_annotations'

# set path for coco json to be saved
save_json_path = 'C:/Users/yz18514/detectron2/Faster_RCNN/data/train.json'

# conert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)