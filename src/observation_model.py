import numpy as np
import math
from skimage import transform as tf
from scipy.signal import convolve2d
from skimage.measure import block_reduce
from random import randint


def random_tf_matrix(translation_range, rotation_range):
    """
        Generate a random homogeneous transformation matrix within the provided ranges

        Returns
        -------
        The transformation matrix in the form

            [[cos(theta),     -sin(theta),  tx]
            [ sin(theta),     cos(theta),   ty]
            [0,               0,            1]]
        Where tx, ty is the translation in pixels and theta is the rotation in radians
    """

    theta = randint(rotation_range[0], rotation_range[1])
    theta = math.radians(theta)

    tx = randint(translation_range[0], translation_range[1])
    ty = randint(translation_range[0], translation_range[1])

    return np.array([
        [math.cos(theta), -math.sin(theta), tx],
        [math.sin(theta), math.cos(theta), ty],
        [0, 0, 1]
    ])


def random_low_res(image, n, psf, downsample, translation_range, rot_range, noise_scale):
    """

    Parameters
    ----------
    image : ndarray
        The original high-resolution image
    n : Integer
        The number of low-resolution images to generate
    psf : ndarray
        Point spread function to convolve with the image for blur
    downsample : Integer
        Factor to down-sample the hr image
    translation_range : tuple
        Min and max value in pixels that the random translation can be selected from
    rot_range : tuple
        Min and max value that the random rotation angle can be in degrees (counter-clockwise)
    noise_scale : float
        Variance of the added gaussian noise

    Returns
    -------

    """
    low_res, tf_matrices = [], []

    for index in range(n):
        im = image

        # Random motion in the form of translation and rotation
        transform_mat = random_tf_matrix(translation_range, rot_range)
        tform = tf.EuclideanTransform(transform_mat)
        im = tf.warp(im, tform)

        # Blur with the PSF
        im = convolve2d(im, psf, 'same')

        # Down-sample the image by averaging local blocks
        im = block_reduce(im, (downsample, downsample), func=np.mean)

        # Add gaussian noise. Default mean 0 variance
        noise = np.random.normal(size=im.shape, scale=noise_scale)
        im += noise

        low_res.append(im)
        tf_matrices.append(transform_mat)

    return low_res, tf_matrices


def normalize(im, new_min, new_max):
    """
        Linear normalization to convert image to any range of values

                        (new_max-new_min)
        In = (I-Min) * ___________________ + new_min
                           max - min

    """
    return (im-im.min()) * (new_max - new_min) / (im.max() - im.min()) + new_min
