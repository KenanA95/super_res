import numpy as np
import math
from low_resolution import LowResolution
from skimage import transform as tf
from scipy.signal import convolve2d
from skimage.measure import block_reduce
from skimage import data
from matplotlib import pyplot as plt


class ObservationModel:

    def __init__(self, image, n, psf, downsample_factor, translation_range, rotation_range, noise_scale):
        self.image = image
        self.psf = psf
        self.downsample = downsample_factor
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.noise_scale = noise_scale
        self.low_resolution = []
        self.add_low_resolution(n)

    def add_low_resolution(self, n):

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

        :param translation_range: Min and max value in pixels that the random translation can be
        :param rotation_range: Min and max value that the random rotation angle can be (counter-clockwise)
        :return:
            [[cos(theta),     -sin(theta),  tx]
            [ sin(theta),     cos(theta),   ty]
            [0,               0,            1]]

        """

        theta = np.random.randint(self.rotation_range[0], self.rotation_range[1])
        theta = math.radians(theta)
        tx, ty = np.random.randint(self.translation_range[0], self.translation_range[1], size=(1, 2))[0]

        return np.array([
            [math.cos(theta), -math.sin(theta), tx],
            [math.sin(theta), math.cos(theta),  ty],
            [0, 0, 1]
        ])


if __name__ == "__main__":

    hr = data.camera()

    psf = np.ones((7, 7)) / 7 ** 2

    camera = ObservationModel(hr, n=10, psf=psf, downsample_factor=4, noise_scale=0.03, translation_range=(-15, 15),
                              rotation_range=(-3, 3))

    plt.imshow(camera.low_resolution[0].image, cmap='gray')
    plt.show()
