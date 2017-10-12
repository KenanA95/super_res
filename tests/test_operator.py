import unittest
from linear_operator import decimation_matrix
from skimage import data


class TestOperator(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_decimation(self):
        # 512x512 original image
        im = data.camera()

        # Decimate the image by a factor of 4
        dec_mat = decimation_matrix(im.shape[0], im.shape[1], downsample_factor=4)
        im = dec_mat * im.flat

        # Result should be 128x128 represented as a vector
        self.assertEqual(len(im), 128**2)


if __name__ == '__main__':
    unittest.main()
