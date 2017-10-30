import unittest
import numpy as np
from observation_model import normalize
from matplotlib import pyplot as plt
from linear_operator import blur_matrix, decimation_matrix, transformation_matrix
import scipy.sparse as sparse
from iterative import gradient_descent, lsqr_restore, irani_peleg_restore
from skimage.measure import compare_mse

__doc__ = """
Test the iterative reconstruction methods where the translation and PSF are known. Verify that the solution is
within a reasonable margin of error.

Note: The answer will never be the exact same because there will always be information loss during
the sub-sampling of the image. There are also other factors like the geometric transform applied by observation
model
"""


class TestIterative(unittest.TestCase):

    def setUp(self):
        # Starting 100x100 high-res image
        self.im = plt.imread("lenna.png")

        # Create a set low-res images with no translations and a spatially-invariant PSF
        self.psf = np.ones((3, 3)) / 9
        n = 10

        # Create a set of random transformation matrices
        self.transform_matrices = []
        for i in range(n):

            # Represent just translation
            tx, ty = np.random.randint(-3, 3, size=(1, 2))[0]
            tf_mat = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            self.transform_matrices.append(np.linalg.inv(tf_mat))

        # Generate the low-res observations using the operator
        self.A = construct_operator(self.transform_matrices, 100, 100, self.psf, downsample_factor=2)
        self.low_res = self.A * self.im.flat
        self.low_res = np.reshape(self.low_res, (n*50, 50))
        self.low_res = np.vsplit(self.low_res, n)

    def test_lsqr(self):
        restored = lsqr_restore(self.low_res, self.A, x0=np.zeros((100, 100)), iter_lim=None, damp=0)
        im = normalize(self.im, 0, 1).astype(float)
        restored = normalize(restored, 0, 1).astype(float)
        mse = compare_mse(im, restored)
        # Averaged squared error < 0.009. In a scale from 0-255 that's an averaged squared error approx. < 2 DN
        self.assertLess(mse, 9e-3)


# Construct an operator that also accounts for the PSF. When it comes to the  NavCam images we don't need to
# because the high-resolution target is the PSF, but its necessary for any other set of images
def construct_operator(tf_matrices, M, N, psf, downsample_factor):

    operators = []
    for tf in tf_matrices:

        D = decimation_matrix(M, N, downsample_factor)
        F = transformation_matrix(tf, (M, N))
        H = blur_matrix(M, N, psf)
        operators.append(D * F * H)

    return sparse.vstack(operators, format='csr')


if __name__ == '__main__':
    unittest.main()
