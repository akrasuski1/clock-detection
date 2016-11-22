import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, utils
from skimage import io, filters, transform, morphology, color, measure, draw, data
import sys, glob, copy, random
import math

def detect_hand_color(arg_image, m, minlen, ax, row, col, r_m, c_m):
    bwm = m[:, :, 0]
    image = filters.gaussian(arg_image, sigma=3, multichannel=True)
    circled2 = m * image
    ax[row][col].imshow(circled2, cmap=plt.cm.gray)
    pixels = np.reshape(circled2[r_m, c_m], (len(r_m), 3))
    #print("pixels shape: ", pixels.shape)
    # Using only 1000 random pixels for color quantization.
    sample = utils.shuffle(pixels)[:1000]
    kmeans = cluster.KMeans(n_clusters=2).fit(sample)
    #clusters = kmeans.cluster_centers_
    labels = kmeans.predict(pixels)
    #print(labels.shape)
    predicted = np.zeros_like(bwm)
    predicted[r_m, c_m] = labels

    whites = sum(sum(predicted))
    blacks = sum(sum((1 - predicted) * bwm))
    #print ("W", whites, "B", blacks)
    if whites > blacks:
        #print ("Flip")
        predicted = bwm * (1 - predicted)

    ax[row][col+1].imshow(predicted, cmap=plt.cm.gray)
    hands = arg_image * color.gray2rgb(predicted)

    lines = transform.probabilistic_hough_line(
        predicted, threshold=10, line_length=minlen, line_gap=3)
    ax[row][col+2].imshow(circled2)
    ax[row][col+3].imshow(np.ones_like(original))
    for line in lines:
        p0, p1 = line
        print line
        ax[row][col+2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[row][col+3].plot((p0[0], p1[0]), (p0[1], p1[1]))
        angle=math.atan2(p1[1]-p0[1],p1[0]-p0[0])*180/math.pi
        if angle<0:
            angle+=180
        print angle
    print("END OF PHOTO")

def redraw(colors, labels, w, h, d):
    image = np.zeros((w, h, d))
    k = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = colors[labels[k]]
            k += 1
    return image


def check_avg_dist(img):
    s = 0.0
    w, h = img.shape[:2]
    for i in range(1000):
        c1 = img[random.randint(0, h - 1)][random.randint(0, w - 1)]
        c2 = img[random.randint(0, h - 1)][random.randint(0, w - 1)]
        c1 -= c2
        s += np.sqrt(c1[0] ** 2 + c1[1] ** 2 + c1[2] ** 2)
    return s / 1000


#MASKA
to_mask = data.astronaut()
to_mask = transform.resize(to_mask, (512, 512, 3))
to_mask = np.zeros_like(to_mask)
r_mask, c_mask = draw.circle(255.5, 255.5, 200)
to_mask[r_mask, c_mask] = 1

if len(sys.argv) > 1:
    clocks = sys.argv[1:]
else:
    clocks = sorted(glob.glob("clocks/*"))
for s in clocks:
    print(s)
    image = io.imread(s) / 255.0
    orig_rows = image.shape[0]
    orig_cols = image.shape[1]
    orig = copy.copy(image)
    image = transform.resize(image, (512, 512))
    original = copy.copy(image)
    image = color.rgb2xyz(image)
    avg_dist = check_avg_dist(image)
    print (avg_dist)

    fig, ax = plt.subplots(9, 9, figsize=(27, 27))
    ax[0][0].imshow(image)
    w, h, d = image.shape
    pixels = np.reshape(image, (w * h, d))
    # Using only 1000 random pixels for color quantization.
    sample = utils.shuffle(pixels)[:1000]
    kmeans = cluster.KMeans(n_clusters=8).fit(sample)
    labels = kmeans.predict(pixels)
    real_colors = kmeans.cluster_centers_
    cnt = 1
    potentialclocks = []
    for col in real_colors:
        out = color.rgb2gray(np.zeros_like(image))
        for i in range(10):
            diff = image - col
            diff = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2
            out += diff < i / 20.0
        ax[0][cnt].imshow(out, cmap=plt.cm.hot)

        diff = image - col
        diff = diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2
        out = diff < (0.1 + avg_dist / 5)  # Around 0.2 works fine too.
        ax[1][cnt].imshow(out, cmap=plt.cm.gray)
        out = morphology.remove_small_holes(out, 64)
        ax[2][cnt].imshow(out, cmap=plt.cm.gray)
        out = morphology.opening(out, morphology.disk(5))
        ax[3][cnt].imshow(out, cmap=plt.cm.gray)
        out = morphology.closing(out, morphology.disk(5))
        ax[4][cnt].imshow(out, cmap=plt.cm.gray)
        out = morphology.remove_small_holes(out, 4096)
        out = morphology.remove_small_objects(out, 1024)
        ax[5][cnt].imshow(out, cmap=plt.cm.gray)
        out = morphology.opening(out, morphology.disk(13))
        ax[6][cnt].imshow(out, cmap=plt.cm.gray)
        components, no = morphology.label(out, return_num=True, background=0)
        ax[7][cnt].imshow(components, cmap=plt.cm.gist_ncar)
        found = color.rgb2gray(np.zeros_like(image)) > 1
        for i in range(1, no + 1):
            comp = (components == i)
            hull = morphology.convex_hull_image(comp)
            orig_area = float(sum(sum(comp)))
            hull_area = float(sum(sum(hull)))
            if orig_area / hull_area > 0.85 and hull_area> 0.01 * w * h:
                #print(hull_area)
                found = found | hull
                potentialclocks.append(hull)
        ax[8][cnt].imshow(found, cmap=plt.cm.gray)

        cnt += 1

    plt.savefig("layers/" + s.split("/")[-1])
    plt.close()
    fig_fin, ax_fin = plt.subplots(16, 16, figsize=(30, 20))
    for id, img in enumerate(potentialclocks):
        components, no = morphology.label(img, return_num=True, background=0)
        props = measure.regionprops(components)
        up = props[0].bbox[0]
        left = props[0].bbox[1]
        down = props[0].bbox[2]
        right = props[0].bbox[3]
        xx = props[0].centroid[1]
        yy = props[0].centroid[0]
        y_r = (down - up)/2
        x_r = (right - left)/2
        r = min(props[0].equivalent_diameter/2.5, 511 - xx, xx, 511 - yy, yy)
        circle = np.zeros_like(original)
        rr, cc = draw.ellipse(yy, xx, y_r, x_r)
        if max(rr) < 512 and min(rr) >= 0 and max(cc) < 512 and min(cc) >= 0:
            circle[rr, cc] = 1
            imagee = filters.gaussian(original, sigma=3, multichannel=True)
            circled = circle * imagee
            ax_fin[int(id/2)][(id%2)*8].imshow(circled, cmap=plt.cm.gray)
            ax_fin[int(id / 2)][(id % 2) * 8 + 1].imshow(orig, cmap=plt.cm.gray)
            up_orig = int(up * orig_rows / 511)
            down_orig = int((down * orig_rows / 511))
            left_orig = int(left * orig_cols / 511)
            right_orig = int((right * orig_cols / 511))
            #print("bbox original: ", up_orig, left_orig, down_orig, right_orig)
            ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((left_orig, right_orig), (up_orig, up_orig), 'r-')
            ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((left_orig, right_orig), (down_orig, down_orig), 'r-')
            ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((left_orig, left_orig), (up_orig, down_orig), 'r-')
            ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((right_orig, right_orig),(up_orig, down_orig),  'r-')
            to_norm = orig[up_orig:down_orig, left_orig:right_orig]
            to_detect = transform.resize(to_norm, (512, 512))
            ax_fin[int(id / 2)][(id % 2) * 8 + 2].imshow(to_norm, cmap=plt.cm.gray)
            ax_fin[int(id / 2)][(id % 2) * 8 + 3].imshow(to_detect, cmap=plt.cm.gray)

            detect_hand_color(to_detect, to_mask, to_mask.shape[0]/5.5, ax_fin, int(id / 2), (id % 2) * 8 + 4, r_mask, c_mask)
    plt.savefig("finals/" + s.split("/")[-1])
    plt.close()

