import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy
import cv2
import os
import glob


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


img_dir = r"E:\PyProjects\Report\T-sne\N2\Image"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []
for num, f1 in enumerate(files):
    img = cv2.imread(f1)
    img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
    deg = numpy.random.randint(-10,10)
    img = rotate_image(img,deg)
    img = cv2.resize(img, (60, 60))
    data.append(img.reshape(-1))

L = len(data)



# img_dir = r"E:\PyProjects\Report\T-sne\data7\R"  # Enter Directory of all images
# data_path = os.path.join(img_dir, '*g')
# files = glob.glob(data_path)
# for num, f1 in enumerate(files):
#     img = cv2.imread(f1)
#     img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
#     deg = numpy.random.randint(-10, 10)
#     img = rotate_image(img, deg)
#     img = cv2.resize(img, (60, 60))
#     data.append(img.reshape(-1))


tsne = TSNE(n_components=2, angle=0.1, perplexity=10, verbose=1, learning_rate=200.0, n_iter=2000)
tsne_results = tsne.fit_transform(data)

plt.plot(tsne_results[:L, 0], tsne_results[:L, 1], 'bo')
plt.plot(tsne_results[L+1:,0],tsne_results[L+1:,1],'ro')
plt.show()
