import numpy as np
from scipy import ndimage
from scipy.ndimage import fourier_shift
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt


def calculate_centroid(im):
    thresh = threshold_otsu(im)
    binary = im > thresh
    return ndimage.measurements.center_of_mass(binary)


# Align the stars according to their centroids
def centroid_align(stars):

    center = int(stars[0].shape[0] / 2), int(stars[0].shape[1] / 2)
    aligned = []
    for star in stars:
        centroid = calculate_centroid(star)

        shift = np.subtract(center, centroid)
        star = fourier_shift(np.fft.fftn(star), shift)
        star = np.fft.ifftn(star).real
        aligned.append(star)

    return aligned
