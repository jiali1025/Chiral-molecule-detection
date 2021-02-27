# Chiral Molecule-Detection

## What's It
Chiral Molecule-Detection is a one-image-one-system deep learning framework that implements state-of-the-art deep learning machine vision algorithm, a uniquely designed data selection method, and effective data augmentation techniques to provide a versatile solution capable of the automated recognition of a complex SPM pattern.

## Workflow
![image](https://user-images.githubusercontent.com/65342604/109371168-a1597780-78de-11eb-98e5-fad59e7dc505.png)

## Pre-requirements
Python ≥ 3.6     
https://www.python.org/ but install conda is recommended, due to it provides more packages in build, you don’t have to install packages one by one.

Pytorch ≥ 1.4    
https://pytorch.org/ click get stated, select your preference, and run the install command based on your working environment.

Numpy ≤ 1.17   
Install by using pip install or conda. Don’t install NumPy 1.18 because it cannot safely interpret flaot64 as an integer. You’ll receive this kind of error when evaluating.

## Installation
Install detectron2, imgaug, and labelIMG. Install detectron2 on Windows is quite time-consuming. Mac OS is recommended if you got one. If you want to install detectron2 on Windows, please refer to useful link.
Install detectron2: https://github.com/facebookresearch/detectron2
Install imgaug: https://github.com/aleju/imgaug
Install labelIMG: https://github.com/tzutalin/labelImg

## Getting Started
See [DEMO.md](./DEMO.md)

## User Interface
See [UI.md](./UI.md)
