import numpy as np
import math
from skimage import transform as tf
from scipy.signal import convolve2d
from skimage.measure import block_reduce


class ObservationModel:
    """
        Given a high-resolution image, this module generates low-resolution representations

        Parameters
        ----------
        image : ndarray
            The original high-resolution image
        n : Integer
            The number of low-resolution images to generate
        psf : ndarray
            Point spread function to convolve with the image for blur
        downsample_factor : Integer
            Factor to down-sample the hr image
        translation_range : tuple
            Min and max value in pixels that the random translation can be selected from
        rotation_range : tuple
            Min and max value that the random rotation angle can be in degrees (counter-clockwise)
        noise_scale : float
            Variance in the added gaussian noise

    """
    def __init__(self, image, n, psf, downsample_factor, translation_range, rotation_range, noise_scale):
        self.image = image
        self.psf = psf
        self.downsample = downsample_factor
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.noise_scale = noise_scale
        self.low_resolution = []
        self.generate_low_resolution(n)

    def generate_low_resolution(self, n):

        for index in range(n):

            im = self.image

            # Random motion in the form of translation and rotation
            transform_mat = self.transformation_matrix()
            tform = tf.EuclideanTransform(transform_mat)
            im = tf.warp(im, tform)

            # Blur with the PSF
            im = convolve2d(im, self.psf, 'same')

            # Down-sample the image by averaging local blocks
            im = block_reduce(im, (self.downsample, self.downsample), func=np.mean)

            # Add gaussian noise. Default mean 0 variance
            noise = np.random.normal(size=im.shape, scale=self.noise_scale)
            im += noise

            # Add to the data-set
            lr = LowResolution(im, transform_mat, self.downsample)
            self.low_resolution.append(lr)

    def transformation_matrix(self):
        """
            Generate a homogeneous transformation matrix with random translation/rotation within the provided range

            Returns
            -------
            The transformation matrix in the form

                [[cos(theta),     -sin(theta),  tx]
                [ sin(theta),     cos(theta),   ty]
                [0,               0,            1]]

            Where tx, ty is the translation in pixels and theta is the rotation in radians
        """

        theta = np.random.randint(self.rotation_range[0], self.rotation_range[1])
        theta = math.radians(theta)
        tx, ty = np.random.randint(self.translation_range[0], self.translation_range[1], size=(1, 2))[0]

        return np.array([
            [math.cos(theta), -math.sin(theta), tx],
            [math.sin(theta), math.cos(theta),  ty],
            [0, 0, 1]
        ])


class LowResolution:
    """
        Store the low-resolution image and the information unique to it i.e the random transformation
        matrix and random noise

    """
    def __init__(self, image, transform_matrix, noise):
        self.image = image
        self.transform_matrix = transform_matrix
        self.noise = noise


def normalize(im, new_min, new_max):
    """
        Linear normalization to convert image to any range of values

                        (new_max-new_min)
        In = (I-Min) * ___________________ + new_min
                           max - min

    """
    return (im-im.min()) * (new_max - new_min) / (im.max() - im.min()) + new_min

