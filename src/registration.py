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
def cross_corr_align(low_res):
    src = low_res[0]

    aligned = []
    for lr in low_res:
        shift, error, diffphase = register_translation(src, lr, 100)
        im = fourier_shift(np.fft.fftn(lr), shift)
        im = np.fft.ifftn(im).real
        im = normalize(im, new_min=0, new_max=1)
        aligned.append(im)

    return aligned


def estimate_tf_matrices(low_res):
    """
        Estimate the translations of each star according the center of the image

        Parameters
        ----------
        low_res: list
            List of the low-resolution frames

        Returns
        -------
        transform_matrices: list
            (3, 3) transformation matrices representing translation from the center of the image
    """
    center = int(low_res[0].shape[0] / 2), int(low_res[0].shape[1] / 2)
    transform_matrices = []

    for lr in low_res:
        # Calculate how far off the centroid is from the center of the image
        ty, tx = calculate_centroid(lr)
        tx, ty = np.subtract((tx, ty), center)
        tf = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        transform_matrices.append(tf)

    return transform_matrices
