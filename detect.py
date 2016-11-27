import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, utils
from skimage import io, filters, transform, morphology, color, measure, draw, data
import sys, glob, copy, random
import math

def distance(a, b):
    y = b[1]-a[1]
    x = b[0]-a[0]
    ans= y**2 + x**2
    return ans**(0.5)

def get_line_angle(line):
    p0,p1=line
    ap=math.atan2(p1[1]-p0[1],p1[0]-p0[0])*180/math.pi
    if ap<0:
        ap+=180
    return ap

def get_line_angle_diff(seg1, seg2, doAbs=True):
    ap=get_line_angle(seg1)
    ar=get_line_angle(seg2)
    diff=ap-ar
    if doAbs:
        diff=abs(diff)
    if diff>90:
        diff=180-diff
    return diff

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(AB,CD):
    A=AB[0]
    B=AB[1]
    C=CD[0]
    D=CD[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def distance_point_line(mid, seg):
    x1=seg[0][0]
    y1=seg[0][1]
    x2=seg[1][0]
    y2=seg[1][1]
    x0=mid[0]
    y0=mid[1]

    num=abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 )
    den=math.sqrt( (y2-y1)**2 + (x2-x1)**2 )

    return num/den

def are_segments_similar(seg1, seg2):
    diff=get_line_angle_diff(seg1, seg2)
    if diff>10: # 10 degrees of difference.
        return False
    # Okay, now we still need to test their shift from origin.
    p0,p1=seg1
    r0,r1=seg2
    mid=[ (p0[i]+p1[i]+r0[i]+r1[i])/4 for i in [0,1] ]
    dp=distance_point_line(mid, seg1)
    dr=distance_point_line(mid, seg2)
    sm=dp+dr
    if sm*(diff+5)>200: # "Distance" between them should not be bigger than some threshold,
        # but if they are similar angle-wise, then lower the threshold
        return False
    return True

def dfs(i, components, lines, used):
    for j,line in enumerate(lines):
        if are_segments_similar(line, lines[i]):
            if j not in used:
                used+=[j]
                components[-1]+=[j]
                dfs(j, components, lines, used)

def are_watches_similar(w, v):
    dist=math.hypot(w[3][0]-v[3][0], w[3][1]-v[3][1])
    rad=(w[2]+v[2])/2
    # If hands intersect in both clocks in almost the same place, these are the same clocks.
    return dist < rad*0.5

def dfsw(i, components, watches, used):
    for j,w in enumerate(watches):
        if are_watches_similar(w, watches[i]):
            if j not in used:
                used+=[j]
                components[-1]+=[j]
                dfsw(j, components, watches, used)

def detect_hand_color(arg_image, m, minlen, ax, row, col, r_m, c_m, uldr):
    bwm = m[:, :, 0]
    image = filters.gaussian(arg_image, sigma=1, multichannel=True)
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
    for i,line in enumerate(lines):
        p0, p1 = line
        ax[row][col+2].plot((p0[0], p1[0]), (p0[1], p1[1]))

    components=[]
    used=[]
    for i in range(len(lines)):
        if i not in used:
            components+=[[i]]
            used+=[i]
            dfs(i, components, lines, used)
    #print (components)
    for i,comp in enumerate(components):
        for line in comp:
            p0, p1 = lines[line]
            ax[row][col+3].plot((p0[0], p1[0]), (p0[1], p1[1]), color=("rgbcmyk"*999)[i])
    if len(components)>6:
        return [] # This one is no good.
    possible_hands=[]
    for i,comp in enumerate(components):
        ok=True
        for line in comp:
            for line2 in comp:
                if get_line_angle_diff(lines[line], lines[line2])>25:
                    # If this ball of lines has too many angles, skip it.
                    ok=False
        if ok:
            sum_ang=0
            for line in comp:
                ang=get_line_angle_diff(lines[comp[0]], lines[line], doAbs=False)
                if ang<-90:
                    ang+=180
                if ang>90:
                    ang-=90
                sum_ang+=ang
            ang=get_line_angle(lines[comp[0]])+sum_ang/len(comp)
            if ang<0:
                ang+=180
            if ang>180:
                ang-=180
            direction=(math.cos(ang*math.pi/180), math.sin(ang*math.pi/180))
            furthest=lines[comp[0]][0]
            closest=lines[comp[0]][0]
            maxi=np.dot(direction, closest)
            mini=maxi
            for line in comp:
                for p in lines[line]:
                    dot=np.dot(p, direction)
                    if dot>maxi:
                        furthest=p
                        maxi=dot
                    if dot<mini:
                        closest=p
                        mini=dot
            p0, p1 = furthest, closest
            ax[row][col+1].plot((p0[0], p1[0]), (p0[1], p1[1]), color=("rgbcmyk"*999)[i])
            def decode(p, uldr):
                def dec(x, u, d):
                    return u+(x/512.0)*(d-u)
                res=(dec(p[0],uldr[1],uldr[3]), dec(p[1], uldr[0], uldr[2]))
                return res
            possible_hands+=[(decode(p0,uldr), decode(p1,uldr))]
    possible_watches=[]
    for id1, hand in enumerate(possible_hands):
        for id2, hand2 in enumerate(possible_hands):
            if id1 < id2:
                if intersect(hand, hand2):
                    pos=line_intersection(hand, hand2)
                    radius=0
                    bad=False
                    for h in [hand, hand2]:
                        for p in h:
                            r=math.hypot(p[0]-pos[0], p[1]-pos[1])
                            if r>radius:
                                radius=r
                    for h in [hand, hand2]:
                        p=h[0]
                        r=math.hypot(p[0]-pos[0], p[1]-pos[1])
                        p=h[1]
                        s=math.hypot(p[0]-pos[0], p[1]-pos[1])
                        ratio=r/s
                        if ratio>1:
                            ratio=1.0/ratio
                        if ratio>0.7: # If a hand is divided by the other hand roughly in half, it's not a clock.
                            bad=True

                    if bad:
                        continue # Intersection at weird place.
                    possible_watches.append((hand, hand2, radius, pos, uldr))
    #print("END OF PHOTO")
    return possible_watches

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

def contrast_of(watch, image, ax, ind):
    # Calculate average color of clock hands and color of background
    handmask=np.zeros_like(image)[:, :, 0]
    h0=watch[0]
    h1=watch[1]
    rr,cc=draw.line(int(h0[0][1]), int(h0[0][0]), int(h0[1][1]), int(h0[1][0]))
    handmask[rr,cc]=1
    rr,cc=draw.line(int(h1[0][1]), int(h1[0][0]), int(h1[1][1]), int(h1[1][0]))
    handmask[rr,cc]=1
    handmask=morphology.dilation(handmask, morphology.disk(3))
    hm=sum(sum(handmask))
    hand_color=[sum(sum(image[:, :, i]*handmask))/hm for i in range(3)]
    if ind<16:
        ax[14][ind].imshow(handmask, cmap=plt.cm.hot)
        ax[15][ind].imshow(color.gray2rgb(morphology.dilation(handmask,morphology.disk(17)))*image)

    circlemask=np.zeros_like(handmask)
    rr,cc=draw.circle(watch[3][0], watch[3][1], watch[2])
    rr=np.clip(rr, 0, 511)
    cc=np.clip(cc, 0, 511)
    circlemask[rr,cc]=1
    cm=sum(sum(circlemask))
    circle_color=[sum(sum(image[:, :, i]*circlemask))/cm for i in range(3)]

    diff = [hc-cc for hc, cc in zip(hand_color, circle_color)]
    diff = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
    return math.sqrt(diff)

#MASKA
to_mask = data.astronaut()
to_mask = transform.resize(to_mask, (512, 512, 3))
to_mask = np.zeros_like(to_mask)
to_end = np.ones_like(to_mask[:,:,0])
r_mask, c_mask = draw.circle(255.5, 255.5, 200)
to_mask[r_mask, c_mask] = 1

fw=open("foundclocks","w")
if len(sys.argv) > 1:
    clocks = sys.argv[1:]
else:
    clocks = sorted(glob.glob("clocks/*"))
for s in clocks:
    print(s)
    fw.write("\n"+s+"\n")
    image = io.imread(s) / 255.0
    orig_rows = image.shape[0]
    orig_cols = image.shape[1]
    orig = copy.copy(image)
    image = transform.resize(image, (512, 512))
    original = copy.copy(image)
    image = color.rgb2xyz(image)
    avg_dist = check_avg_dist(image)

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
                found = found | hull
                potentialclocks.append(hull)
        ax[8][cnt].imshow(found, cmap=plt.cm.gray)

        cnt += 1

    plt.savefig("layers/" + s.split("/")[-1])
    plt.close()
    fig_fin, ax_fin = plt.subplots(16, 16, figsize=(30, 20))
    watches=[]
    for id, img in enumerate(potentialclocks):
        components, no = morphology.label(img, return_num=True, background=0)
        props = measure.regionprops(components)
        up = props[0].bbox[0]
        left = props[0].bbox[1]
        down = props[0].bbox[2]
        right = props[0].bbox[3]
        uldr=props[0].bbox
        xx = props[0].centroid[1]
        yy = props[0].centroid[0]
        y_r = (down - up)/2
        x_r = (right - left)/2
        r = min(props[0].equivalent_diameter/2.5, 511 - xx, xx, 511 - yy, yy)
        circle = np.zeros_like(original)
        rr, cc = draw.ellipse(yy, xx, y_r, x_r)
        rr=np.clip(rr, 0, 511)
        cc=np.clip(cc, 0, 511)
        circle[rr, cc] = 1
        imagee = filters.gaussian(original, sigma=3, multichannel=True)
        circled = circle * imagee
        ax_fin[int(id/2)][(id%2)*8].imshow(circled, cmap=plt.cm.gray)
        ax_fin[int(id / 2)][(id % 2) * 8 + 1].imshow(orig, cmap=plt.cm.gray)
        up_orig = int(up * orig_rows / 511)
        down_orig = int((down * orig_rows / 511))
        left_orig = int(left * orig_cols / 511)
        right_orig = int((right * orig_cols / 511))
        ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((left_orig, right_orig), (up_orig, up_orig), 'r-')
        ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((left_orig, right_orig), (down_orig, down_orig), 'r-')
        ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((left_orig, left_orig), (up_orig, down_orig), 'r-')
        ax_fin[int(id / 2)][(id % 2) * 8 + 1].plot((right_orig, right_orig),(up_orig, down_orig),  'r-')
        to_norm = orig[up_orig:down_orig, left_orig:right_orig]
        to_detect = transform.resize(to_norm, (512, 512))
        ax_fin[int(id / 2)][(id % 2) * 8 + 2].imshow(to_norm, cmap=plt.cm.gray)
        ax_fin[int(id / 2)][(id % 2) * 8 + 3].imshow(to_detect, cmap=plt.cm.gray)

        watches+=detect_hand_color(to_detect, to_mask, 100.0, ax_fin, int(id / 2), (id % 2) * 8 + 4, r_mask, c_mask, uldr)
    #print ("Original watches:")
    contrasts=[]

    components=[]
    used=[]
    true_uldr = []
    for i in range(len(watches)):
        true_uldr.append([0,0,0,0])
        if i not in used:
            components+=[[i]]
            used+=[i]
            dfsw(i, components, watches, used)
    #print (components)

    norm_watches=[]

    for comp in components:
        new_uldr=[512.0,512.0,0.0,0.0]
        for c in comp:
            if watches[c][4][0] < new_uldr[0]:
                new_uldr[0] = watches[c][4][0]
            if watches[c][4][1] < new_uldr[1]:
                new_uldr[1] = watches[c][4][1]
            if watches[c][4][2] > new_uldr[2]:
                new_uldr[2] = watches[c][4][2]
            if watches[c][4][3] > new_uldr[3]:
                new_uldr[3] = watches[c][4][3]
        for c in comp:
            true_uldr[c]=new_uldr

    #print(true_uldr)

    for i,watch in enumerate(watches):
        contrasts.append(contrast_of(watch, image, ax_fin, i))

        #print (watch[2], watch[3], watch[0], watch[1], contrasts[-1])
        #print ("ULDR: ", watch[4])
        #print("TRUE_ULDR: ", true_uldr[i])
        x00 = ((watch[0][0][0]-true_uldr[i][1])*512.0)/(true_uldr[i][3]-true_uldr[i][1])
        y00 = ((watch[0][0][1]-true_uldr[i][0])*512.0)/(true_uldr[i][2]-true_uldr[i][0])
        x01 = ((watch[0][1][0]-true_uldr[i][1])*512.0)/(true_uldr[i][3]-true_uldr[i][1])
        y01 = ((watch[0][1][1]-true_uldr[i][0])*512.0)/(true_uldr[i][2]-true_uldr[i][0])
        x10 = ((watch[1][0][0]-true_uldr[i][1])*512.0)/(true_uldr[i][3]-true_uldr[i][1])
        y10 = ((watch[1][0][1]-true_uldr[i][0])*512.0)/(true_uldr[i][2]-true_uldr[i][0])
        x11 = ((watch[1][1][0]-true_uldr[i][1])*512.0)/(true_uldr[i][3]-true_uldr[i][1])
        y11 = ((watch[1][1][1]-true_uldr[i][0])*512.0)/(true_uldr[i][2]-true_uldr[i][0])
        l2 = watch[2]
        x3 = ((watch[3][0]-true_uldr[i][1])*512.0)/(true_uldr[i][3]-true_uldr[i][1])
        y3 = ((watch[3][1]-true_uldr[i][0])*512.0)/(true_uldr[i][2]-true_uldr[i][0])
        newwatch = (((x00,y00),(x01,y01)),((x10,y10),(x11,y11)),l2,(x3,y3))
        #print(newwatch[2], newwatch[3], newwatch[0], newwatch[1])
        norm_watches.append(newwatch)

    threshold_contrast=0
    maksik=[]
    for comp in components:
        makss=0.0
        for c in comp:
            if makss < distance(norm_watches[c][0][0], norm_watches[c][0][1]):
                makss = distance(norm_watches[c][0][0], norm_watches[c][0][1])
            if makss < distance(norm_watches[c][1][0], norm_watches[c][1][1]):
                makss = distance(norm_watches[c][1][0], norm_watches[c][1][1])
        maksik.append(makss)
    print(maksik)
    all_times=[]
    for ident, comp in enumerate(components):
        uldr_=true_uldr[comp[0]]
        #print ("Component:",comp)
        final_hands=[]
        for c in comp:
            to_check=[True, True]
            if len(final_hands) < 2 and distance(norm_watches[c][0][0],norm_watches[c][0][1]) * 2.0 > maksik[ident]:
                final_hands.append(norm_watches[c][0])
                to_check[0] = False
            if len(final_hands) < 2 and distance(norm_watches[c][1][0], norm_watches[c][1][1]) * 2.0 > maksik[ident]:
                final_hands.append(norm_watches[c][1])
                to_check[1] = False
            for index, hand in enumerate(final_hands):
                if to_check[0] and distance(norm_watches[c][0][0],norm_watches[c][0][1]) * 2.0 < maksik[ident]:
                    to_check[0]=False
                if to_check[1] and distance(norm_watches[c][1][0],norm_watches[c][1][1]) * 2.0 < maksik[ident]:
                    to_check[1]=False
                if to_check[0] and are_segments_similar(norm_watches[c][0], final_hands[index]):
                    if distance(norm_watches[c][0][0],norm_watches[c][0][1]) > distance(final_hands[index][0],final_hands[index][1]):
                        final_hands[index] = norm_watches[c][0]
                    to_check[0]=False
                if to_check[1] and are_segments_similar(norm_watches[c][1], final_hands[index]):
                    if distance(norm_watches[c][1][0],norm_watches[c][1][1]) > distance(final_hands[index][0],final_hands[index][1]):
                        final_hands[index] = norm_watches[c][1]
                    to_check[1]=False
            if to_check[0]==True:
                    final_hands.append(norm_watches[c][0])
            if to_check[1]==True:
                    final_hands.append(norm_watches[c][1])

        ax_fin[13][ident].imshow(np.ones_like(to_end), cmap=plt.cm.gray)
        for h in final_hands:
            ax_fin[13][ident].plot((h[0][0],h[1][0]), (h[0][1], h[1][1]), 'r-')

        ang=0
        avg_cent=[255,255]
        for iden1, h1 in enumerate(final_hands):
            for iden2, h2 in enumerate(final_hands):
                if iden1 < iden2:
                    newang = get_line_angle_diff(h1, h2, doAbs=True)
                    if newang>ang:
                        ang = newang
                        avg_cent[0], avg_cent[1] = line_intersection(h1, h2)
        wskazowki=[]
        dl=[]
        for h in final_hands:
            if distance(h[0],avg_cent) > distance(h[1],avg_cent):
                wskazowki.append((avg_cent, h[0]))
                dl.append(distance(avg_cent, h[0]))
            else:
                wskazowki.append((avg_cent, h[1]))
                dl.append(distance(avg_cent, h[1]))

        clock_dim=
        if len(wskazowki)==2:
            if dl[0]>dl[1]:
                godz = wskazowki[1]
                minn = wskazowki[0]

            else:
                godz = wskazowki[0]
                minn = wskazowki[1]
            fin_min = 90+math.atan2(minn[1][1]-minn[0][1],minn[1][0]-minn[0][0])*180/math.pi
            fin_godz = 90+math.atan2(godz[1][1]-godz[0][1],godz[1][0]-godz[0][0])*180/math.pi
            if fin_min < 0:
                fin_min += 360
            if fin_godz < 0:
                fin_godz += 360

            fin_min /= 6
            fin_min = int(round(fin_min))
            if fin_min == 60:
                fin_min = 0

            fin_godz /= 30
            fin_godz = int(round(fin_godz))
            if fin_min > 30:
                fin_godz -= 1
            if fin_godz == 12:
                fin_godz = 0
            if fin_godz == -1:
                fin_godz = 11
            
            all_times.append(fin_godz, fin_min, 0, uldr_, avg_cent)
            print("Possible Time: "+str(fin_godz) + ":" + str(fin_min) + ":00")
            fw.write("Possible Time: "+str(fin_godz)+":"+str(fin_min)+":00"+"\n")
            #print("min", fin_min)
            #print("godz", fin_godz)

        elif len(wskazowki)>=3:
            maks3 = [0,0,0]
            sekid = -1
            minid = -1
            godzid = -1
            for kol, i in enumerate(wskazowki):
                if maks3[0]<dl[kol]:
                    maks3[2] = maks3[1]
                    maks3[1] = maks3[0]
                    maks3[0] = dl[kol]
                    godzid = minid
                    minid = sekid
                    sekid = kol
                elif maks3[1]<dl[kol]:
                    maks3[2] = maks3[1]
                    maks3[1] = dl[kol]
                    godzid = minid
                    minid = kol
                elif maks3[2]<dl[kol]:
                    maks3[2] = dl[kol]
                    godzid = kol
            sek = wskazowki[sekid]
            minn = wskazowki[minid]
            godz = wskazowki[godzid]
            fin_sek = 90+math.atan2(sek[1][1]-sek[0][1],sek[1][0]-sek[0][0])*180/math.pi
            fin_min = 90+math.atan2(minn[1][1]-minn[0][1],minn[1][0]-minn[0][0])*180/math.pi
            fin_godz = 90+math.atan2(godz[1][1]-godz[0][1],godz[1][0]-godz[0][0])*180/math.pi
            if fin_sek < 0:
                fin_sek += 360
            if fin_min < 0:
                fin_min += 360
            if fin_godz < 0:
                fin_godz += 360

            fin_min /= 6
            fin_min = int(round(fin_min))
            if fin_min == 60:
                fin_min = 0

            fin_godz /= 30
            fin_godz = int(round(fin_godz))
            if fin_min > 30:
                fin_godz -= 1
            if fin_godz == 12:
                fin_godz = 0
            if fin_godz == -1:
                fin_godz = 11

            fin_sek /= 6
            fin_sek = int(round(fin_sek))
            if fin_sek == 60:
                str_sek = 0

            all_times.append(fin_godz, fin_min, 0, uldr_, avg_cent)
            print("Possible Time: "+str(fin_godz)+":"+str(fin_min)+":"+str(fin_sek))
            fw.write("Possible Time: "+str(fin_godz)+":"+str(fin_min)+":"+str(fin_sek)+"\n")
            #print("sek", fin_sek)
            #print("min", fin_min)
            #print("godz", fin_godz)

        else:
            print("Can't match to clock")



    print all_times
    fw.flush()
    plt.savefig("finals/" + s.split("/")[-1])
    plt.close()

