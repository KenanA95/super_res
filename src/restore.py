import numpy as np
from registration import centroid_align
from scipy import misc
from observation_model import normalize
from skimage import restoration


def interpolation_restore(low_res, psf, downsample_factor, align_function=centroid_align,):

    # Align the stars according to their centroids
    low_res = align_function(low_res)

    # Average all of the low-resolution frames
    average_image = np.mean(low_res, axis=0)

    # Up-sample through bicubic interpolation
    high_res = misc.imresize(average_image, downsample_factor*100, interp='bicubic')

    # Deconvolve the PSF
    high_res = normalize(high_res, new_min=0, new_max=1)
    high_res = restoration.richardson_lucy(high_res, psf, iterations=5)

    return high_res

