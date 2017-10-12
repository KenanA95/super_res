import unittest
from linear_operator import decimation_matrix, blur_matrix
from skimage import data
import numpy as np
from scipy.signal import convolve2d
import numpy.testing as npt


class TestOperator(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

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

if __name__ == '__main__':
    unittest.main()
