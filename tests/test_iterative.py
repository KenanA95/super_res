import unittest
import numpy as np
from skimage import data
from observation_model import ObservationModel
from scipy.ndimage import fourier_shift
from matplotlib import pyplot as plt
from iterative import gradient_descent, lsqr_restore, irani_peleg_restore


# 1. Create a set of low-resolution images with the exact translations and psf known
# 2. Reconstruct the original image
# 3. Make sure the two are within a small margin

# Note: The answer will never be the exact same because there will always be information loss during
# the sub-sampling of the image. There are also other factors such as what kind of geometric transform is used


class TestIterative(unittest.TestCase):

    def setUp(self):
        # Starting 512x512 high-res image
        im = data.camera()

        # Create a set low-res images with no translations and a spatially-invariant PSF
        psf = np.ones((3, 3)) / 9
        low_res = ObservationModel(im, n=15, psf=psf, downsample_factor=4, translation_range=(0, 0),
                                   rotation_range=(0, 0), noise_scale=0).low_res_images

        # Add a set of known translations
        self.transform_matrices = []
        self.low_res = []
        for lr in low_res:

            # Transform matrix to represent just translation
            tx, ty = np.random.randint(-20, 20, size=(1, 2))[0]
            tf_mat = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            lr = fourier_shift(np.fft.fftn(lr), (ty, tx))
            lr = np.fft.ifftn(lr).real
            self.low_res.append(lr)
            self.transform_matrices.append(tf_mat)

    def tearDown(self):
        self.low_res = []
        self.transform_matrices = []

    def test_lsqr(self):
        pass

if __name__ == '__main__':
    unittest.main()
