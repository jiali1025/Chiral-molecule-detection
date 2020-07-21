# From tkinter
import sys, os
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import tkinter.filedialog as fd
import time
import threading
import re

# From detectron2
#import random
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
#import train_data
#import cv2
#from detectron2.engine import DefaultTrainer
#from detectron2.config import get_cfg
#import os
#from detectron2.engine.defaults import DefaultPredictor
#from detectron2.utils.visualizer import ColorMode
#import matplotlib
#import matplotlib.pyplot as plt


def uploadFile():
    global path
    path = fd.askopenfilename()
    print(path)

def run_HPB():
    str = "python C:/Users/yz18514/detectron2/Faster_RCNN_HPB/predictor.py --input %s"%path
    print(str)
    os.system(str)

def run_FHPB():
    str = "python C:/Users/yz18514/detectron2/Faster_RCNN_FHPB/predictor.py --input %s"%path
    print(str)
    os.system(str)

def thread_it(func, *args):
    t = threading.Thread(target=func, args=args) 
    t.setDaemon(True) 
    t.start()

top = tk.Tk()
top.iconbitmap('C:/Users/yz18514/detectron2/icon.ico')

def center_window(w, h):
    ws = top.winfo_screenwidth()
    hs = top.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    top.geometry('%dx%d+%d+%d' % (w, h, x, y))
center_window(600, 600)

top.title("Instance Segmentation")
# 背景
canvas = tk.Canvas(top, width=600, height=600, bd=0, highlightthickness=0,bg='White')
imgpath = 'C:/Users/yz18514/Desktop/Background1.gif'
imgpath2 = 'C:/Users/yz18514/Desktop/Background2.gif'
img = Image.open(imgpath)
img2 = Image.open(imgpath2)
photo = ImageTk.PhotoImage(img)
photo2 = ImageTk.PhotoImage(img2)
canvas.create_image(150,300,image=photo)
canvas.create_image(450,300,image=photo2)
canvas.pack()


label = tk.Label(top, text="Molecular Chirality Recognition",width=100,height=50,font=('Times New Roman', 20),fg='DimGray',bg='White')
canvas.create_window(300, 50, width=400, height=50,
                     window=label)

#label2 = tk.Label(top, text="Hexaphenylbenzene",width=100,height=50,font=('Times New Roman', 18),fg='DarkGray',bg='White')
#canvas.create_window(150, 100, width=200, height=50,
#                     window=label2)

#label3 = tk.Label(top, text="F-Hexaphenylbenzene",width=100,height=50,font=('Times New Roman', 18),fg='DarkGray',bg='White')
#canvas.create_window(450, 100, width=200, height=50,
#                     window=label3)


btn = tk.Button(text="Upload Image",command=lambda :thread_it(uploadFile),fg='Black',bg='SlateGray')

canvas.create_window(300, 500, width=200, height=50,
                     window=btn)

btn2 = tk.Button(text="Hexaphenylbenzene \n Click to Start Processing",command=lambda :thread_it(run_HPB),fg='DarkGray',bg='White')

canvas.create_window(150, 125, width=200, height=50,
                     window=btn2)

btn3 = tk.Button(text="F-Hexaphenylbenzene \n Click to Start Processing",command=lambda :thread_it(run_FHPB),fg='DarkGray',bg='White')

canvas.create_window(450, 125, width=200, height=50,
                     window=btn3)

top.mainloop()

