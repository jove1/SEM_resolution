# SEM Resolution

This program is designed to determine SEM resolution from the image of gold nanoparticles on carbon.
[Gaussian profile](https://en.wikipedia.org/wiki/Normal_distribution) is fitted across particle edges, and the width (&sigma;) is reported.


## Instal

Install [Python 3](https://www.python.org/downloads/) according your operating system. 
Then the required libraries:


```
pip3 install -r requirements.txt 
```

## Usage

```
./resolution.py --help
usage: resolution.py [-h] [--crop TOP BOTTOM LEFT RIGHT] [--pixel PIXEL] [--wait] [--interval INTERVAL] [--length LENGTH] [--fraction FRACTION] [FILE [FILE ...]]

Determine SEM resolution.

positional arguments:
  FILE                  *.tif files to process, generates syntetic data if no files are specified

optional arguments:
  -h, --help            show this help message and exit
  --crop TOP BOTTOM LEFT RIGHT
                        crop image margins (default: 0 0 0 0)
  --wait                wait after each image
  --interval INTERVAL   profile interval along contour (default: 10)
  --length LENGTH       profile length (default: 75)
  --fraction FRACTION   fraction of included profiles ordered by fit residual (default: 0.2)
```

## Operation

The histogram is checked for under/over exposed images that are rejected.
The criterium is **less than 1% of pixels in low/high 1/16 of histogram**. This is
indicated by green/red area and vertical lines in output report.

Edges are found by [thresholding in the minimum of the histogram](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_minimum).

[Gaussian cummulative density function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html)
is fitted to profiles perpedicular to the edges (`--length`, `--interval`).

The statistics (mean, standard deviation, median and quartiles) for resulting &sigma; of the best fit profiles are reported (`--fraction`).


## Output

![Example output](output.png)
