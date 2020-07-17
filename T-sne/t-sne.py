import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import cv2
import os
import glob

img_dir = r"Faster_RCNN\T-sne\FHPB\L"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    data.append(img.reshape(-1))

L = len(data)

img_dir = r"Faster_RCNN\T-sne\FHPB\R"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    data.append(img.reshape(-1))


tsne = TSNE(n_components=2, angle=0.1, perplexity=10, verbose=1, learning_rate=200.0, n_iter=4000)
tsne_results = tsne.fit_transform(data)

plt.plot(tsne_results[:L, 0], tsne_results[:L, 1], 'bo')
plt.plot(tsne_results[L+1:,0],tsne_results[L+1:,1],'ro')
plt.show()
