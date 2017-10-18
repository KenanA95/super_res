import numpy as np
from registration import centroid_align
from scipy import misc
from observation_model import ObservationModel, normalize
from registration import calculate_centroid
from scipy.ndimage import fourier_shift
from scipy.signal import convolve2d

# Irani-Peleg iterative back-projection method. Work in progress


def upsample_image(im, n):
    """
         Upsample the image to n x the original size of each dimension,
         with the new even rows and columns filled with zeros (0)

    """
    new_shape = np.multiply(im.shape, n)
    upsampled_im = np.zeros(new_shape)
    upsampled_im[0::n, 0::n] = im

    return upsampled_im


def irani_peleg_restore(low_res, psf, downsample_factor, iterations, k=1, align_function=centroid_align):

    # Initial estimate is the upsampled averaged image
    average = np.mean(align_function(low_res), axis=0)
    hr_estimate = misc.imresize(average, downsample_factor*100, interp='bicubic')
    hr_estimate = normalize(hr_estimate, new_min=0, new_max=1)

    for i in range(iterations):
        error = 0

        for lr in low_res:
            center = lr.shape[1] / 2, lr.shape[0] / 2

            # Create a low-res frame with no motion
            simulated_lr = ObservationModel(hr_estimate, n=1, psf=psf, downsample_factor=downsample_factor,
                                            translation_range=(0, 0), rotation_range=(0, 0),
                                            noise_scale=0.001).low_res_images[0]

            # Estimate the offset of the star from the center of the image
            ty, tx = calculate_centroid(lr)
            tx, ty = np.subtract((tx, ty), center)

            # Add the same motion to the simulated LR frame
            simulated_lr = fourier_shift(np.fft.fftn(simulated_lr), (ty, tx))
            simulated_lr = np.fft.ifftn(simulated_lr).real

            delta = simulated_lr - lr
            error += np.sum(abs(delta))

            delta = upsample_image(delta, downsample_factor)
            delta = convolve2d(delta, psf**k, 'same')

            hr_estimate += delta

        print("Iteration: {0} \t Error: {1}".format(i, error))

    return normalize(hr_estimate, new_min=0, new_max=1)