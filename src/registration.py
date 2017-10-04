import numpy as np
from scipy import ndimage
from scipy.ndimage import fourier_shift
from skimage.filters import threshold_otsu
from skimage.feature import register_translation
from observation_model import normalize


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


# Align using phase correlation according to first image in the set
def cross_corr_align(lr_frames):
    src = lr_frames[0]

    aligned = []
    for lr in lr_frames:
        shift, error, diffphase = register_translation(src, lr, 100)
        im = fourier_shift(np.fft.fftn(lr), shift)

        im = np.fft.ifftn(im).real
        im = normalize(im, new_min=0, new_max=1)

        aligned.append(im)

    return aligned
