# A step-by-step demo:

After the software and packages above successfully installed, a simple 1-hour demo to train a Faster R-CNN model for F-HPB system molecule recognition is provided in “FasterR-CNN_demo.zip.” The structure of the demo folder should be: 

FasterR-CNN_demo
├─Database
│  │  Convert.py
│  │  ImgAugmentation.py
│  ├─testset1
│  │      Reference.png
│  │      Test.png
│  └─trainset1
│      ├─Annotations
│      └─Images
│              0.jpg
├─LabelIMG
│  └─labelImg.exe
└─Results
    │  Inference.py
    │  Train.py
└─output
Another folder named “FasterR-CNN_demo_complete,” which contains all files created in this demo, is also provided in the zip file for reference. 

## 1. Select an image for dataset generation.
Put this image in \FasterR-CNN_Demo\Database\trainset1\Images\0.jpg.
The image is already provided in this demo. Create more folders for trainsets or test sets if necessary.

## 2. Label the selected image by using labelImg. 
We need to provide a label (either left-handed or right-handed) for each molecule in the image for Faster R-CNN to learn.
Open \FasterR-CNN_Demo_Complete\LabelIMG\ labelImg.exe. 

![image](https://user-images.githubusercontent.com/65342604/109371633-c64eea00-78e0-11eb-8174-356849cdb7f0.png)

Click the “Open” button and open the image.

![image](https://user-images.githubusercontent.com/65342604/109371638-cbac3480-78e0-11eb-9549-d34ac3604fe8.png)
Draw bounding boxes on molecular instances with correct labels.

![image](https://user-images.githubusercontent.com/65342604/109371641-d1a21580-78e0-11eb-8470-cb7230fde893.png)
After labeling all molecules on the image.
After labeling all molecules on the image.
Save the annotation file as \FasterR-CNN_Demo\Database\trainset1\Annotations\0.xml

## 3. Run file \FasterR-CNN_Demo\Database\Convert.py to convert annotation in 0.xml format to 0.json format.
![image](https://user-images.githubusercontent.com/65342604/109371666-eb435d00-78e0-11eb-91a9-81e0d5af12d8.png)
Key in python command line the path of Convert.py, the directory of 0.xml file, and the path of output 0.json file.

(venv) E:\PyProjects> E:\PyProjects\FasterR-CNN_Demo\Database\Convert.py E:\PyProjects\FasterR-CNN_Demo\Database\trainset1\Annotations E:\PyProjects\FasterR-CNN_Demo\Database\trainset1\Annotations\0.json

Number of xml files: 1
Success: E:\PyProjects\FasterR-CNN_Demo\Database\trainset1\Annotations\0.json

The annotations in json format will be created at \FasterR-CNN_Demo\Database\trainset1\Annotations\0.json

## 4. Run file \FasterR-CNN_Demo\ImgAugmentation.py to generate a training set.

Code you need to rewrite:
ImgAugmentation.py:

![image](https://user-images.githubusercontent.com/65342604/109371694-03b37780-78e1-11eb-8da8-67feab3f2f95.png)

Line 5: Change the directory to the correct directory of the training set.
Augmentation techniques used for dataset generation are already pre-defined in this demo. They are good enough to obtain a Faster R-CNN model for high-resolution SPM images.

Then run ImgAugmentation.py. It will create a dataset of 1000 augmented images at \FasterR-CNN_Demo\Database\trainset1\Images, and also the corresponding annotation file for the dataset at \FasterR-CNN_Demo\Database\trainset1\Annotations\dataset.json

Now, we have prepared a dataset for Faster R-CNN training.

## 5. Run file Train.py to train a Faster R-CNN model on the dataset generated before. 
Code you need to rewrite:

Train.py:
Line 9: Change the directory to the correct directory of the training set.
![image](https://user-images.githubusercontent.com/65342604/109371723-25146380-78e1-11eb-99c4-132b0de96a03.png)
Model hyperparameters and training settings are already pre-defined in this demo. They are good enough to obtain a Faster R-CNN model for high-resolution SPM images.

Then run Train.py. It will usually take less than 30min to complete the training depending on your GPU. The model will be automatically saved at \FasterR-CNN_Demo_Complete\Results\output\model_final.pth after finishing.

## 6. Use the trained model to analyze the molecular patterns on another experimental image by Inference.py.
Code you need to rewrite:
Inference.py:
Line 16: Change the path of the image required to analyze. A sample image is provided in this demo at \FasterR-CNN_Demo_Complete\Database\testset1\Test.png.
Line 19: Change the directory to the correct directory of the training set.
Line 36: Change the path to the path of the automatically saved model model_final.pth.

![image](https://user-images.githubusercontent.com/65342604/109371729-31002580-78e1-11eb-9a0b-dd837236d4a9.png)
Then, run Inference.py. It will generate an inference result in around 10 seconds.

![image](https://user-images.githubusercontent.com/65342604/109371734-378e9d00-78e1-11eb-8ec7-a249eb61b40c.png)
You can compare this result with a reference provided in this demo at \FasterR-CNN_Demo\Database\testset1\Reference.png. A more human-friendly inference result is also saved on your desktop.

![image](https://user-images.githubusercontent.com/65342604/109371740-41b09b80-78e1-11eb-9359-4603d69bb3cc.png)
Notes: The model trained in demo may not be robust enough for images of lower resolution. Please refer to main text and section 6 Computing Materials (Python files) for more information on data augmentation techniques and model hyperparameters to obtain a more robust model.

