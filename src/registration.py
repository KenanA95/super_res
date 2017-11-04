import numpy as np
from scipy import ndimage
from scipy.ndimage import fourier_shift
from skimage.filters import threshold_otsu
from skimage.feature import register_translation


def calculate_centroid(im):
    """ Use Otsu's method to distinguish between star and background and then calculate the centroid """
    thresh = threshold_otsu(im)
    binary = im > thresh
    return ndimage.measurements.center_of_mass(binary)


def centroid_align(stars):
    """ Align the stars according to their centroids  """
    center = int(stars[0].shape[0] / 2), int(stars[0].shape[1] / 2)
    aligned = []

    for star in stars:
        centroid = calculate_centroid(star)
        shift = np.subtract(center, centroid)
        star = fourier_shift(np.fft.fftn(star), shift)
        star = np.fft.ifftn(star).real
        aligned.append(star)

    return aligned


def cross_corr_align(low_res):
    """ Align using phase correlation according to first image in the set """
    src = low_res[0]
    aligned = []

    for lr in low_res:
        shift, error, diffphase = register_translation(src, lr, 100)
        im = fourier_shift(np.fft.fftn(lr), shift)
        im = np.fft.ifftn(im).real
        aligned.append(im)

    return aligned


def est_centroid_translations(low_res):
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

        transform_matrices.append(np.linalg.inv(tf))

    return transform_matrices
