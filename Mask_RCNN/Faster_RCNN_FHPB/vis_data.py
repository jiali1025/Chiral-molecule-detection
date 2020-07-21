import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import mol_data
import cv2



mol_metadata = MetadataCatalog.get("mol")
print(mol_metadata)
dataset_dicts = DatasetCatalog.get("mol")


for d in random.sample(dataset_dicts, 6):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=mol_metadata)
    vis = visualizer.draw_dataset_dict(d)
    print(vis.get_image()[:, :, ::-1])
    img = vis.get_image()[:, :, ::-1]
    cv2.imshow('rr', img)
    cv2.waitKey(0)