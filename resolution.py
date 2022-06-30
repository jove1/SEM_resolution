#!/usr/bin/env python3

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import PIL.Image
import skimage

def make_particles(N=2048, M=200,
    sep=5, size=100, size_sigma=0.5, value_sigma=0.2):

    data = np.zeros((N,N))
    circles = []
    while len(circles) < M:
        x, y = np.random.uniform(0, N, 2)
        r = np.random.lognormal(np.log(size), size_sigma)
        v = np.random.lognormal(0, value_sigma)
        for x1, y1, r1 in circles:
            if (x-x1)**2 + (y-y1)**2 < (r+r1+sep)**2:
                break
        else:
            coords = skimage.draw.disk((x,y), r, shape=data.shape)
            data[coords] = v

            circles.append((x,y,r))
            print(".", end="", flush=True)
    print()
    return data

def kittler_illingworth(h):
    "Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986)."
    i = np.arange(h.size)

    h1 = np.cumsum(h)
    ih1 = np.cumsum(i*h)
    iih1 = np.cumsum(i*i*h)

    # this has precision problems with floats, 
    # we have int histograms though

    h2 = h1[-1]-h1 
    ih2 = ih1[-1]-ih1
    iih2 = iih1[-1]-iih1
    
    with np.errstate(invalid="ignore", divide="ignore"):
        s1 = iih1/h1 - ih1*ih1/h1/h1
        s2 = iih2/h2 - ih2*ih2/h2/h2

        criterion = np.where((h1==0)|(s1==0), 0, h1*np.log(s1/h1/h1)) + \
                    np.where((h2==0)|(s2==0), 0, h2*np.log(s2/h2/h2))

    return np.argmin(criterion), criterion

def resolution(data, label):
    print(label)

    #
    # Crop databar
    #

    data = data[:2048]

    #
    # Check image size and bitdepth
    #

    print(data.shape, data.dtype)
    if np.min(data.shape) < 1024:
        print("image too small")
        return

    if data.dtype != np.uint16:
        print("not a 16-bit image")
        return

    data = data/65535

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2)
    ax = ax.ravel()

    ax[0].set_title(label)
    ax[0].imshow(data, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(data[:512,:512], cmap="gray", vmin=0, vmax=1)

    #
    # Calculate histogram and check over/under exposure
    #

    h, bins = np.histogram(data.ravel(), 128, (0,1) )

    too_dark = h[:8].sum() > 0.01*h.sum()
    too_bright = h[-8:].sum() > 0.01*h.sum()

    bpos = (bins[1:] + bins[:-1])/2
    bwidth = bins[:-1]-bins[1:]
    ax[2].bar(bpos[:8], h[:8], bwidth[:8], color="r" if too_dark else "g")
    ax[2].bar(bpos[8:-8], h[8:-8], bwidth[8:-8], color="C0")
    ax[2].bar(bpos[-8:], h[-8:], bwidth[-8:], color="r" if too_bright else "g")
    ax[2].axvline(bins[8], ls="--", c="k")
    ax[2].axvline(bins[-9], ls="--", c="k")
    ax[2].set_xlim(0,1)

    if too_dark:
        print("too many dark pixels")

    if too_bright:
        print("too many bright pixels")

    if too_dark or too_bright:
        return
  
    #
    # Find optimal threshold for gaussian mixture
    #

    thresh_index, criterion = kittler_illingworth(h) 
    thresh = bins[thresh_index+1]
    ax[2].twinx().plot(bins[1:], criterion, "C1")

    bad_thresh = h[:thresh_index+1].sum() < 0.25*h.sum() or \
                 h[thresh_index+1:].sum() < 0.25*h.sum()

    ax[2].axvline(thresh, ls="--", c="r" if bad_thresh else "g")

    if bad_thresh:
        print("bimodal distribution not found. Too much noise?")
        return

    #
    # Normals to blob contours
    #

    N = 10
    L = 30
    contours = skimage.measure.find_contours(scipy.ndimage.gaussian_filter(data, 3).T, thresh)
    profiles = []
    for c in contours:
        for ca, cb in zip(c[::N][:-1], c[::N][1:]):
            p = (ca+cb)/2 # center
            q = cb-ca # tangent 
            q = np.array([-q[1], q[0]]) # normal
            q /= np.hypot(q[0], q[1]) # normalize
            pa, pb = p+L*q,  p-L*q # endpoints

            profiles.append( (pa, pb) )

    from matplotlib.collections import LineCollection
    c_contours = LineCollection(contours, colors="C0")
    c_profiles = LineCollection(profiles, colors="C1")
    ax[0].add_collection(c_profiles)
    ax[0].add_collection(c_contours)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Determine SEM resolution.')
    parser.add_argument('files', nargs="*", metavar="FILE", help="*.tif files to process")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),  default=(0, 0, 0, 0),
            help="crop image margins (default: 0, 0, 0, 0)")
    parser.add_argument("--pixel", type=float, default=1, help="pixel size (default: 1nm)")
    parser.add_argument("--wait", action="store_true", help="wait after each image")
    args = parser.parse_args()

    if not args.files:    

        base_data = make_particles(2048, 200, 4, 100)
        
        noise = True
        for sigma in [0, 2, 4, 6, (6,2)]:
            data = 0.2 + 0.4 * base_data

            if sigma != 0:
                data = scipy.ndimage.gaussian_filter(data, sigma)
            
            if noise:
                data = np.random.normal(data, 0.02 + 0.05 * np.sqrt(data/0.6))

            data = (data.clip(0, 1) * 65535).astype("uint16")

            resolution(data, "sigma={}".format(sigma))
            print()
            if args.wait:
                plt.show()

        if not args.wait:
            plt.show()

    else:
        for fname in args.files:
            img = PIL.Image.open(fname)

            resolution(np.asarray(img), fname)
            print()
            if args.wait:
                plt.show()

        if not args.wait:
            plt.show()
