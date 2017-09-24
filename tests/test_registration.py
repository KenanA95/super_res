import unittest
import numpy as np
from astropy.convolution import AiryDisk2DKernel
from observation_model import ObservationModel, normalize
from registration import calculate_centroid
from scipy.ndimage import fourier_shift


class TestRegistration(unittest.TestCase):

    def setUp(self):
        # Starting high-res is a 121x121 image of an Airy disk
        hr = np.array(AiryDisk2DKernel(radius=15))
        hr = normalize(hr, new_min=0, new_max=1)

        # Crop to 120x120 to simplify debugging calculations
        hr = hr[0:120, 0:120]

        # Arbitrary PSF
        psf = np.ones((7, 7)) / 7**2

        # Create low-res frames with no motion
        camera = ObservationModel(hr, n=5, psf=psf, downsample_factor=10, translation_range=(0, 0),
                                  rotation_range=(0, 0), noise_scale=0.001)

        self.low_resolution = []
        self.translations = []

        # Add motion to the lr frames and store the exact amount
        for lr in camera.low_resolution:

            # There's no significant rotation in the actual data set only translation
            tx, ty = np.random.randint(-3, 3, size=(1, 2))[0]
            im = fourier_shift(np.fft.fftn(lr.image), (ty, tx))
            im = np.fft.ifftn(im).real

            self.low_resolution.append(im)
            self.translations.append((tx, ty))

    def tearDown(self):
        self.low_resolution = None
        self.translations = None

    def test_registration(self):
        # Center of the image is the reference point
        lr = self.low_resolution[0]
        center = lr.shape[1] / 2, lr.shape[0] / 2

        for index, lr in enumerate(self.low_resolution):
            # Actual x and y translation
            tx, ty = self.translations[index]

            # Estimated offset based on the centroid calculation
            cx, cy = np.subtract(calculate_centroid(lr), center)

            # Make sure they are within a 1/2 pixel margin
            self.assertAlmostEqual(cx, tx, 0)
            self.assertAlmostEqual(cy, ty, 0)

if __name__ == '__main__':
    unittest.main()
