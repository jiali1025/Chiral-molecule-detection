import json
import matplotlib
import numpy
import matplotlib.pyplot as plt

jslists=[]

with open('C:/Users/yz18514/detectron2/Faster_RCNN_HPB/output/metrics.json', 'r') as in_file:
    for jsonObj in in_file:
        jslist = json.loads(jsonObj)
        jslists.append(jslist)

data_time=[]
cls_accuracy=[]
false_negative=[]
fg_cls_accuracy=[]
iteration=[]
loss_box_reg=[]
loss_cls=[]
loss_rpn_cls=[]
loss_rpn_loc=[]
lr=[]
num_bg_samples=[]
num_fg_samples=[]
num_neg_anchors=[]
num_pos_anchors=[]
total_loss=[]

for i, step in enumerate(jslists):
    data_time.append(step["data_time"])
    total_loss.append(step["total_loss"])
    iteration.append(step["iteration"])

total_loss= numpy.array(total_loss)
iteration= numpy.array(iteration)

matplotlib.pyplot.plot(iteration, total_loss)
matplotlib.pyplot.show()