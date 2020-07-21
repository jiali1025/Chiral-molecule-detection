import sys
import os
import glob
import cv2
import numpy as np
import json
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from labelme import utils


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.mkdir(path)
        print('====================')
        print('creat path : ', path)
        print('====================')
    return 0


def check_json_file(path):
    for i in path:
        json_path = i[:-3] + 'json'
        if not os.path.exists(json_path):
            print('error')
            print(json_path, ' not exist !!!')
            sys.exit(1)


def read_jsonfile(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonfile(object, save_path):
    json.dump(object, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def get_points_from_json(json_file):
    point_list = []
    shapes = json_file['shapes']
    for i in range(len(shapes)):
        for j in range(len(shapes[i]["points"])):
            point_list.append(shapes[i]["points"][j])
    return point_list


def write_points_to_json(json_file, aug_points):
    k = 0
    new_json = json_file
    shapes = new_json['shapes']
    for i in range(len(shapes)):
        for j in range(len(shapes[i]["points"])):
            new_point = [aug_points.keypoints[k].x, aug_points.keypoints[k].y]
            new_json['shapes'][i]["points"][j] = new_point
            k = k + 1
    return new_json


# Sequential augumentation choice.
ia.seed(1)

# Define the augmentation pipeline.
sometimes = lambda aug : iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [   
        # Use the following 0 to 5 methods to enhance the image.
        iaa.SomeOf((0, 5),
            [
                # Superpixel representation of some images.
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Use Gaussian Blur, Mean Blur, and Median Blur.
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen.
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Relief effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Edge detection.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add Gaussian noise
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Set 1% to 10% of pixels to black. 
                # Or cover 3% to 15% of the pixels with 2% to 5% black squares of the original image.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # 5% probability of inverting pixel intensity.
                iaa.Invert(0.05, per_channel=True), 

                # Each pixel randomly adds and subtracts a number between -10 and 10.
                iaa.Add((-210, 230), per_channel=0.5),

                # Multiply the pixel by a number between 0.5 or 1.5.
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # Make the contrast of the entire image half or double
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                # Turn RGB into grayscale image and multiply alpha to add to original image
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # Move the pixels to the surrounding area.
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

            ],
            
            random_order=True # Use these operations on images in random order
        )
    ],
    random_order=True         # Use these operations on images in random order
)


if __name__ == '__main__':
    # TO-DO-BELOW (Here is the only place you need to change: two folders path, and number of augmentations)
    # ======================================================================================================
    aug_times = 100
    in_dir = "C:/Users/yz18514/detectron2/Faster_RCNN/data/original_handmade_images_annotations_val"
    out_dir = "C:/Users/yz18514/detectron2/Faster_RCNN/data/val_annotations"
    # ======================================================================================================

    mkdir(out_dir)
    imgs_dir_list = glob.glob(os.path.join(in_dir, '*.png'))
    check_json_file(imgs_dir_list)

    for idx_png_path in imgs_dir_list:
        idx_json_path = idx_png_path[:-3] + 'json'
        idx_img = cv2.imdecode(np.fromfile(idx_png_path, dtype=np.uint8), 1)
        idx_json = read_jsonfile(idx_json_path)
        points_list = get_points_from_json(idx_json)
        kps = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in  points_list], shape=idx_img.shape)

        for idx_aug in range(aug_times):
            image_aug, kps_aug = seq(image=idx_img, keypoints=kps)
            image_aug.astype(np.uint8)
            idx_new_json = write_points_to_json(idx_json, kps_aug)
            idx_new_json["imagePath"] = idx_png_path.split(os.sep)[-1][:-4] + str(idx_aug) + '.png'
            idx_new_json["imageData"] = str(utils.img_arr_to_b64(image_aug), encoding='utf-8')
            new_img_path = os.path.join(out_dir, idx_png_path.split(os.sep)[-1][:-4] + str(idx_aug) + '.png')
            cv2.imwrite(new_img_path, image_aug)
            new_json_path = new_img_path[:-3] + 'json'
            save_jsonfile(idx_new_json, new_json_path)