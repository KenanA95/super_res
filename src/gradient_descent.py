from linear_operator import construct_operator
from matplotlib import pyplot as plt
from registration import centroid_align
import numpy as np
from observation_model import ObservationModel, normalize
from astropy.convolution import AiryDisk2DKernel
from scipy import misc
from restore import compare

# Set up the synthetic data

# Starting high-resolution image 81x81
image = AiryDisk2DKernel(10)

psf = np.ones((3, 3)) / 9

# Create 10 low-resolution frames of 27x27
camera = ObservationModel(image, n=10, psf=psf, downsample_factor=3, translation_range=(-2, 2),
                          rotation_range=(0, 0), noise_scale=0.0)

low_res = camera.low_res_images

# Initial estimate is the averaged image up-sampled through bicubic interpolation
x0 = centroid_align(low_res)
x0 = np.mean(x0, axis=0)
x0 = misc.imresize(x0, 300, 'bicubic')
x0 = normalize(x0, 0, 1)

op = construct_operator(len(low_res), 81, 81, downsample_factor=3, psf=psf)

# Stack all the low-resolution images into a vector
size = len(low_res) * np.prod(low_res[0].shape)
b = np.empty(size)
M = np.prod(low_res[0].shape)

for i in range(len(low_res)):
    b[i * M:(i + 1) * M] = low_res[i].flat


# Apply gradient descent algorithm
damp = 1e-1
x = x0.flat

for i in range(50):
    step = damp * -1 * (op.T * ((op * x) - b))
    print(np.sum(step))
    print("Gradient descent step {0}".format(i))
    x += step


x = np.reshape(x, (81, 81))
plt.imshow(x, cmap='gray')
plt.show()

im = np.array(AiryDisk2DKernel(10))
original = normalize(im, 0, 1)
compare(original, x)