#!/usr/bin/env python3

import numpy as np
import scipy
import matplotlib.pyplot as plt
import PIL.Image
import skimage

def make_particles(img_size=2048, num_particles=200,
    sep=5, size=100, size_sigma=0.5, value_sigma=0.2):

    data = np.zeros((img_size,img_size))
    circles = []
    while len(circles) < num_particles:
        x, y = np.random.uniform(0, img_size, 2)
        r = np.random.lognormal(np.log(size), size_sigma)
        v = np.random.lognormal(0, value_sigma)
        for x1, y1, r1 in circles:
            if (x-x1)**2 + (y-y1)**2 < (r+r1+sep)**2:
                break
        else:
            coords = skimage.draw.disk((x,y), r, shape=data.shape)
            data[coords] = v

            circles.append((x,y,r))
            print(end="\rgenerating synthetic image {:.1f}%".format(100*len(circles)/num_particles), flush=True)
    print()
    return data


def resolution(data, label, options):
    print(label)

    #
    # Crop databar
    #

    data = data[
            options.crop[0]:data.shape[0]-options.crop[1],
            options.crop[2]:data.shape[1]-options.crop[3],
            ]

    #
    # Check image size and bitdepth
    #

    print(data.shape, data.dtype)
    if np.min(data.shape) < 1024:
        print("image too small")
        return

    if data.dtype == np.uint16:
        data = data/65535
    elif data.dtype == np.uint8:
        data = data/255
    else:
        print("not 8-bit or 16-bit image")
        return


    fig = plt.figure(figsize=(10, 7.5))
    ax_image = fig.add_subplot(231)
    ax_zoom = fig.add_subplot(232)
    ax_hist = fig.add_subplot(233)
    ax_fit = fig.add_subplot(234)
    ax_polar = fig.add_subplot(235, polar=True)
    ax_text = fig.add_subplot(236)

    fig.suptitle(label)
    ax_image.imshow(data, cmap="gray", vmin=0, vmax=1)
    ax_zoom.imshow(data[:512,:512], cmap="gray", vmin=0, vmax=1)

    #
    # Calculate histogram and check over/under exposure
    #

    h, bins = np.histogram(data.ravel(), 128, (0,1) )

    too_dark = h[:8].sum() > 0.01*h.sum()
    too_bright = h[-8:].sum() > 0.01*h.sum()

    bpos = (bins[1:] + bins[:-1])/2
    bwidth = bins[:-1]-bins[1:]
    ax_hist.bar(bpos[:8], h[:8], bwidth[:8], color="r" if too_dark else "g")
    ax_hist.bar(bpos[8:-8], h[8:-8], bwidth[8:-8], color="C0")
    ax_hist.bar(bpos[-8:], h[-8:], bwidth[-8:], color="r" if too_bright else "g")
    ax_hist.axvline(bins[8], ls="--", c="k")
    ax_hist.axvline(bins[-9], ls="--", c="k")
    ax_hist.set_xlim(0,1)
    ax_hist.set_yticks([])

    if too_dark:
        print("too many dark pixels")

    if too_bright:
        print("too many bright pixels")

    if too_dark or too_bright:
        return
  
    #
    # Find optimal threshold
    #

    try:
        thresh_index = skimage.filters.threshold_minimum(hist=h)
    except RuntimeError:
        print("bimodal distribution not found. Too much noise?")
        return

    if h[:thresh_index].sum() < 0.1*h.sum():
        print("too small background area")
        return

    if h[thresh_index:].sum() < 0.1*h.sum():
        print("too small particle area")
        return

    thresh = bins[thresh_index]
    ax_hist.axvline(thresh, ls="--", c="C1")

    #
    # Normals to blob contours
    #

    N = options.interval
    L = options.length
    contours = skimage.measure.find_contours(scipy.ndimage.gaussian_filter(data, 3).T, thresh)
    profiles = []
    normals = []
    for c in contours:
        for ca, cb in zip(c[::N][:-1], c[::N][1:]):
            p = (ca+cb)/2 # center
            q = cb-ca # tangent 
            q = np.array([-q[1], q[0]]) # normal
            q /= np.hypot(q[0], q[1]) # normalize
            pa, pb = p+L/2*q,  p-L/2*q # endpoints
            normals.append(q)
            profiles.append( (pa, pb) )
    normals = np.array(normals)
    profiles = np.array(profiles)

    #
    # Fit profile shape
    #

    from scipy.special import ndtr as stepf
    _sqrt2pi = np.sqrt(2*np.pi)
    dstepf = lambda x: np.exp(-x*x/2)/_sqrt2pi

    def f(par, x, y):
        a, b, m, s = par
        return a + (b-a) * stepf( (x-m)/s ) - y

    def df(par, x, y):
        a, b, m, s = par
        arg = (x-m)/s
        ret = np.empty((4, x.size))
        stepf(arg, out=ret[1])
        ret[0] = 1-ret[1]
        ret[2] = (a-b)/s*dstepf(arg)
        ret[3] = ret[2]*arg
        return ret

    def profile_line(data, a, b, N):
        x = np.linspace(a[0], b[0], N, dtype=int).clip(0, data.shape[1]-1)
        y = np.linspace(a[1], b[1], N, dtype=int).clip(0, data.shape[0]-1)
        return data[y, x]

    results = []
    residuals = []
    x = np.arange(L)
    
    num = len(profiles)
    print("{} candicate profiles".format(num))
    for i, prof in enumerate(profiles,1):
        # this is nice but very slow
        #y = skimage.measure.profile_line(data.T, prof[0], prof[1], order=0, mode="nearest")
        y = profile_line(data, prof[0], prof[1], L)

        par0 = [ y[:5].mean(), y[-5:].mean(), L/2, 5]
        res = scipy.optimize.leastsq(f, x0=par0, args=(x, y), Dfun=df, col_deriv=True, full_output=True)
        par = res[0]

        results.append(par)
        residuals.append( np.mean(f(par, x, y)**2) )
        print(end="\rfitting {:.1f}%".format(100*i/num), flush=True)
    print()
    results = np.array(results)
    residuals = np.array(residuals)
    
    #
    # Present results
    #
    m = results[:,2]
    s = np.abs( results[:,3] )
    t = np.arctan2(normals[:,1], normals[:,0])
    
    # exclude bogus fits
    residuals[m>L] = np.inf
    residuals[m<0] = np.inf

    idx = np.argsort(residuals)[:int(len(residuals)*options.fraction)]

    for i in idx:
        y = profile_line(data, profiles[i][0], profiles[i][1], L)
        ax_fit.plot(x, y, c="C0", lw=0.2)
        ax_fit.plot(x, f(results[i], x, 0), c="C1", lw=0.2)
    ax_fit.grid(True)

    from matplotlib.collections import LineCollection
    c_profiles = LineCollection(profiles[idx], colors="C1")
    c_contours = LineCollection(contours, colors="C0")

    ax_image.add_collection(c_profiles)
    ax_image.add_collection(c_contours)

    n_total = len(s)
    t = t[idx]
    s = s[idx]
    n_included = len(s)

    ax_polar.plot(t, s, ".", zorder=0, ms=3)
    ax_polar.set_rlim(0, 2*np.mean(s))


    stat_labels = ["mean", "std", "25%", "median", "75%"]
    stats = np.array([ np.mean(s), np.std(s), *np.percentile(s, [25,50,75]) ])
    factors = [0.77, 1, 1.35, 2]
    factor_labels = [ "{:.2f}σ\n({:.1f}%-\n{:.1f}%)".format(x, 100*stepf(-x/2), 100*stepf(x/2)) for x in factors]

    report = "Statistics for {} best profiles out of total {}.\n".format(n_included, n_total)

    row = [""] + factor_labels
    table = [row]
    for s, sl in zip(stats, stat_labels):
        row = [sl]
        for x in factors:
            row.append("{:.2f}".format(s*x))
        table.append(row)

    import texttable
    t = texttable.Texttable(0)
    #t.set_deco(0)
    t.set_cols_dtype(["t"]*(len(factors)+1))
    t.add_rows(table)

    report += t.draw()
    print(report)

    ax_text.axis("off")
    ax_text.text(0.5, 0.5, report, font="monospace", ha="center", multialignment="left", va="center", transform=ax_text.transAxes, size='small')

    fig.tight_layout()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Determine SEM resolution.')
    parser.add_argument('files', nargs="*", metavar="FILE", help="*.tif files to process, generates syntetic data if no files are specified")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),  default=(0, 0, 0, 0),
            help="crop image margins (default: 0 0 0 0)")
    #parser.add_argument("--pixel", type=float, default=1, help="pixel size (default: 1nm)")
    parser.add_argument("--wait", action="store_true", help="wait after each image")
    parser.add_argument("--interval", type=int, default=10, help="profile interval along contour (default: 10)")
    parser.add_argument("--length", type=int, default=75, help="profile length (default: 75)")
    parser.add_argument("--fraction", type=float, default=0.2, help="fraction of included profiles ordered by fit residual (default: 0.2)")

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

            resolution(data, "sigma={}".format(sigma), args)
            print()
            if args.wait:
                plt.show()

        if not args.wait:
            plt.show()

    else:
        for fname in args.files:
            img = PIL.Image.open(fname)

            resolution(np.asarray(img), fname, args)
            print()
            if args.wait:
                plt.show()

        if not args.wait:
            plt.show()
