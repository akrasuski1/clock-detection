import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, utils
from skimage import io, filters, transform, morphology, color, measure
import sys, glob, copy, random, colorsys

def redraw(colors, labels, w, h, d):
    image = np.zeros((w, h, d))
    k = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = colors[labels[k]]
            k += 1
    return image

def check_avg_dist(img):
    s=0.0
    w,h=img.shape[:2]
    for i in range(1000):
        c1=img[random.randint(0,h-1)][random.randint(0,w-1)]
        c2=img[random.randint(0,h-1)][random.randint(0,w-1)]
        c1-=c2
        s+=np.sqrt(c1[0]**2+c1[1]**2+c1[2]**2)
    return s/1000

if len(sys.argv)>1:
    clocks=sys.argv[1:]
else:
    clocks=sorted(glob.glob("clocks/*"))
for s in clocks:
    print(s)
    image = io.imread(s)/255.0
    image = transform.resize(image, (512,512))
    image=color.rgb2xyz(image)
    avg_dist=check_avg_dist(image)
    print avg_dist

    fig, ax = plt.subplots(9, 9, figsize=(27,27))
    ax[0][0].imshow(image)
    w,h,d = image.shape
    pixels = np.reshape(image, (w * h, d))
    # Using only 1000 random pixels for color quantization.
    sample = utils.shuffle(pixels)[:1000]
    kmeans = cluster.KMeans(n_clusters=8).fit(sample)
    labels = kmeans.predict(pixels)
    real_colors = kmeans.cluster_centers_

    cnt=1
    for col in real_colors:
        out=color.rgb2gray(np.zeros_like(image))
        for i in range(10):
            diff=image-col
            diff=diff[:,:,0]**2 + diff[:,:,1]**2 + diff[:,:,2]**2
            out+=diff<i/20.0
        ax[0][cnt].imshow(out, cmap=plt.cm.hot)

        diff=image-col
        diff=diff[:,:,0]**2 + diff[:,:,1]**2 + diff[:,:,2]**2
        out=diff<(0.1+avg_dist/5) # Around 0.2 works fine too.
        ax[1][cnt].imshow(out, cmap=plt.cm.gray)
        out=morphology.remove_small_holes(out, 64)
        ax[2][cnt].imshow(out, cmap=plt.cm.gray)
        out=morphology.opening(out, morphology.disk(5))
        ax[3][cnt].imshow(out, cmap=plt.cm.gray)
        out=morphology.closing(out, morphology.disk(5))
        ax[4][cnt].imshow(out, cmap=plt.cm.gray)
        out=morphology.remove_small_holes(out, 4096)
        out=morphology.remove_small_objects(out, 1024)
        ax[5][cnt].imshow(out, cmap=plt.cm.gray)
        out = morphology.opening(out, morphology.disk(13))
        ax[6][cnt].imshow(out, cmap=plt.cm.gray)
        
        components, no = morphology.label(out, return_num=True, background=0)
        ax[7][cnt].imshow(components, cmap=plt.cm.gist_ncar)
        
        found=color.rgb2gray(np.zeros_like(image))>1
        for i in range(1, no+1):
            comp=(components==i)
            hull=morphology.convex_hull_image(comp)
            orig_area=float(sum(sum(comp)))
            hull_area=float(sum(sum(hull)))
            if orig_area/hull_area>0.85:
                found=found|hull
        ax[8][cnt].imshow(found, cmap=plt.cm.gray)


        

        cnt+=1
    
    plt.savefig("layers/"+s.split("/")[-1])
    plt.close()
