import unittest
from linear_operator import decimation_matrix, blur_matrix, transformation_matrix
from skimage import data
import numpy as np
from scipy.signal import convolve2d
import numpy.testing as npt
from skimage import transform as tf
from observation_model import normalize
from matplotlib import pyplot as plt


class TestOperator(unittest.TestCase):

    def test_decimation(self):
        # 512x512 original image
        im = data.camera()

        # Decimate the image by a factor of 4
        dec_operator = decimation_matrix(im.shape[0], im.shape[1], downsample_factor=4)
        im = dec_operator * im.flat

        # Result should be 128x128 represented as a vector
        self.assertEqual(len(im), 128**2)

    def test_blur(self):

        # 200x200 image and spatially-invariant psf
        im = data.checkerboard()
        psf = np.ones((5, 5)) / 5**2

        # Use the operator to convolve
        blur_operator = blur_matrix(im.shape[0], im.shape[1], psf)
        im_blurred = blur_operator * im.flat
        im_blurred = np.reshape(im_blurred, (200, 200))

        # Get the actual zero boundary convolution
        actual = convolve2d(im, psf, 'same', boundary='fill', fillvalue=0)

        npt.assert_equal(np.rint(im_blurred), np.rint(actual))

    def test_transform(self):
        im = data.camera()

        # Apply a translation to an image
        tx, ty = 25, 15
        H = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
        tform = tf.EuclideanTransform(H)
        actual_translated = tf.warp(im, tform.inverse)

        # Apply the same translation using the operator
        tf_operator = transformation_matrix(H, im.shape)
        op_translated = tf_operator * im.flat
        op_translated = np.reshape(op_translated, (512, 512))
        op_translated = normalize(op_translated, 0, 1)

        npt.assert_equal(actual_translated, op_translated)

if __name__ == '__main__':
    unittest.main()
