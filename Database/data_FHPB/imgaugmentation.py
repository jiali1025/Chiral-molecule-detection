import os
import imgaug.augmenters as iaa
import json
import cv2


mode = "train"  # Perform augmentation for dataset
qty = 200  # Size of dataset after augmentation


# ================================================================================== #

path = r"E:\PyProjects\Report\Faster_RCNN/data/" + mode + "set"
images = cv2.imread(os.path.join(path, "Images/0.jpg"))
images = images.reshape([1, images.shape[0], images.shape[1], 3])

with open(os.path.join(path, "0.json"), "r") as infile:
    annos = json.load(infile)

bbs0 = [anno["bbox"] for anno in annos["annotations"]]
id0 = annos["images"][0]

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-15, 15),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            fit_output=True
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((2, 3),
                   [
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5)),  # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images

                       iaa.Invert(0.05, per_channel=0.5),  # invert color channels
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-30, 30)),  # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Add((200, 220), per_channel=True),
                           iaa.Add((-150, 10), per_channel=True),
                           iaa.Multiply((0.25, 4.0), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-3, 0),
                               first=iaa.Multiply((0.25, 4.0), per_channel=0.5),
                               second=iaa.LinearContrast((0.25, 4.0))
                           )
                       ]),
                       iaa.LinearContrast((0.25, 4.0), per_channel=0.5),  # improve or worsen the contrast
                       sometimes(iaa.ElasticTransformation(alpha=(0.25, 4.0), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                   ],
                   random_order=True
                   ),
        iaa.Grayscale(alpha=1.0)
    ],
    random_order=False
)


def covert_bbx(bbx_c):
    x0 = bbx_c[0]
    y0 = bbx_c[1]
    w = bbx_c[2]
    h = bbx_c[3]
    return [x0, y0, x0 + w, y0 + h]


def in_covert_bbx(bbx_c):
    x1 = bbx_c[0]
    y1 = bbx_c[1]
    x2 = bbx_c[2]
    y2 = bbx_c[3]
    return [x1, y1, x2 - x1, y2 - y1]


c = len(annos["annotations"])
for i in range(1, qty):

    bbs_aug = [covert_bbx(onebbx) for onebbx in bbs0]
    bbs_aug = [(one[0], one[1], one[2], one[3]) for one in bbs_aug]
    images_aug, bbs_aug = seq(images=images,
                              bounding_boxes=bbs_aug)  # add segment for segmentaion e.g. polygons = polygons
    images_aug = images_aug[0]
    bbs_aug = [in_covert_bbx(onebbx) for onebbx in bbs_aug]

    cv2.imwrite(path + r"\Images\\" + str(i) + ".jpg", images_aug)

    id = id0.copy()
    id["file_name"] = str(i) + ".jpg"
    id["id"] = i
    id["height"] = images_aug.shape[0]
    id["width"] = images_aug.shape[1]
    annos["images"].append(id)

    for n, one in enumerate(bbs_aug):
        int_one = []
        for number in one:
            int_one.append(int(number))
        bbs = annos["annotations"][0].copy()
        bbs["image_id"] = i
        c = c + 1
        bbs["id"] = c

        bbs["bbox"] = int_one
        bbs["area"] = int_one[2] * int_one[3]
        bbs["category_id"] = annos["annotations"][n]["category_id"]
        annos["annotations"].append(bbs)


with open(os.path.join(path, "trainset.json"), 'w') as json_file:
    json.dump(annos, json_file, indent="")
