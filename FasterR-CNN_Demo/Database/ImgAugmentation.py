import imgaug.augmenters as iaa
import json
import cv2

dataset_directory = r"E:\PyProjects\FasterR-CNN_Demo\Database\trainset1"  # Change dataset directory here
image = cv2.imread(dataset_directory + r"\Images\0.jpg")  # Location of the image for dataset generation
Num = 1000  # Number of augmented images
with open(dataset_directory + r"\Annotations\0.json", "r") as infile:  # Location of .json annotation file of the image
    annos = json.load(infile)

#  Augmented images will be stored in "Images" folder
#  Dataset annotations will be store in "Annotation" folder as dataset.json
# =========================================================================================================== #


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
            fit_output=True
        )),
        # execute 2 to 3 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((1, 3),
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
                           iaa.Add((-150, 150), per_channel=True),
                           iaa.Multiply((0.3, 3.0), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-3, 0),
                               first=iaa.Multiply((0.3, 3.0), per_channel=0.5),
                               second=iaa.LinearContrast((0.3, 3.0))
                           )
                       ]),
                       iaa.LinearContrast((0.3, 3.0), per_channel=0.5),  # improve or worsen the contrast
                       sometimes(iaa.ElasticTransformation(alpha=(0.3, 3.0), sigma=0.25)),
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


image = image.reshape([1, image.shape[0], image.shape[1], -1])
bbs0 = [anno["bbox"] for anno in annos["annotations"]]
id0 = annos["images"][0]
c = len(annos["annotations"])

for i in range(1, Num):

    bbs_aug = [covert_bbx(onebbx) for onebbx in bbs0]
    bbs_aug = [(one[0], one[1], one[2], one[3]) for one in bbs_aug]
    images_aug, bbs_aug = seq(images=image,
                              bounding_boxes=bbs_aug)  # add segment for segmentaion e.g. polygons = polygons
    images_aug = images_aug[0]
    bbs_aug = [in_covert_bbx(onebbx) for onebbx in bbs_aug]

    cv2.imwrite(dataset_directory + r"\\Images\\" + str(i) + ".jpg", images_aug)

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

with open(dataset_directory + r"\Annotations\dataset.json", 'w') as json_file:
    json.dump(annos, json_file, indent="")
