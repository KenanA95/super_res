import unittest
import numpy as np
from astropy.convolution import AiryDisk2DKernel
from observation_model import ObservationModel, normalize
from matplotlib import pyplot as plt


class TestRegistration(unittest.TestCase):

    def setUp(self):
        hr = np.array(AiryDisk2DKernel(radius=16))
        hr = normalize(hr, new_min=0, new_max=1)
        psf = np.ones((7, 7)) / 7**2

        camera = ObservationModel(hr, n=5, psf=psf, downsample_factor=10, translation_range=(-15, 15),
                                  rotation_range=(0, 0.001), noise_scale=0.001)

        self.translations = []

        # Get the actual translations
        for lr in camera.low_resolution:

            tx = lr.transform_matrix[0, 2]
            ty = lr.transform_matrix[1, 2]
            print(tx, ty)
            self.translations.append((tx, ty))

    def tearDown(self):
        self.translations = None

    def test_registration(self):
        pass

if __name__ == '__main__':
    unittest.main()
