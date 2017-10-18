from linear_operator import construct_operator
from registration import centroid_align
import numpy as np
from observation_model import ObservationModel, normalize
from astropy.convolution import AiryDisk2DKernel
from scipy import misc
from restore import compare


def gradient_descent(low_res, psf, x0, upsample_factor, iterations, damp=1e-1):

    # Stack all the low-resolution images into the vector b in lexicographical order
    size = len(low_res) * np.prod(low_res[0].shape)
    b = np.empty(size)
    M = np.prod(low_res[0].shape)

    for i in range(len(low_res)):
        b[i * M:(i + 1) * M] = low_res[i].flat

    #
    M, N = low_res[0].shape[0] * upsample_factor, low_res[0].shape[1] * upsample_factor
    A = construct_operator(len(low_res), M, N, downsample_factor=upsample_factor, psf=psf)

    x = x0.copy().flat

    for i in range(iterations):
        step = damp * -1 * (A.T * ((A * x) - b))
        prior = damp * np.subtract(x, x0.flat)
        x += step  # + prior
        print("Gradient descent step {0}".format(i))

    return np.reshape(x, (M, N))

# EXAMPLE

# Starting high-resolution image 145x145
image = AiryDisk2DKernel(10)

psf = np.ones((3, 3)) / 9

# Create 10 low-resolution frames of 29x29
camera = ObservationModel(image, n=15, psf=psf, downsample_factor=3, translation_range=(-2, 2),
                          rotation_range=(0, 0), noise_scale=0.0)

low_res = camera.low_res_images

# Initial estimate is the averaged image up-sampled through bicubic interpolation
x0 = centroid_align(low_res)
x0 = np.mean(x0, axis=0)
x0 = misc.imresize(x0, 300, 'bicubic')
x0 = normalize(x0, 0, 1)

original = np.array(AiryDisk2DKernel(10))
original = normalize(original, 0, 1)

x = gradient_descent(low_res, psf, x0, upsample_factor=3, iterations=10)
compare(original, x)
