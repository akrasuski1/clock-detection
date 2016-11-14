import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, utils
from skimage import io, filters, transform, morphology, color
import colorsys
import sys
import glob

for s in sorted(glob.glob("clocks/*")):
    print s
    image = io.imread(s)/255.0
    image = transform.resize(image, (512,512))
    w,h,d = image.shape

    pixels = np.reshape(image, (w * h, d))
    # Using only 1000 random pixels for color quantization.
    sample = utils.shuffle(pixels)[:1000]
    kmeans = cluster.KMeans(n_clusters=8).fit(sample)
    labels = kmeans.predict(pixels)
    real_colors = kmeans.cluster_centers_ # Unused?

    def redraw(colors, labels, w, h, d):
        image=np.zeros((w, h, d))
        k=0
        for i in range(w):
            for j in range(h):
                image[i][j]=colors[labels[k]]
                k+=1
        return image

    fig, ax = plt.subplots(3, 4, figsize=(8,4))
    ax[0][0].imshow(image)
    # High contrast colors
    colors=[(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    quant=redraw(colors, labels, w, h, d)
    ax[1][0].imshow(quant)
    quant_true=redraw(real_colors, labels, w, h, d)
    ax[2][0].imshow(quant_true)
    for i,col in enumerate(colors):
        one=color.rgb2gray(quant==col)>0.99
        one=morphology.closing(one, morphology.disk(3))
        one=morphology.opening(one, morphology.disk(3))
        ax[i%3][1+i/3].imshow(one, cmap=plt.cm.gray)
    plt.savefig("out/kmeans/"+s.split("/")[-1])
