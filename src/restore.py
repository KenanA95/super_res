import numpy as np
from registration import centroid_align
from scipy import misc
from observation_model import ObservationModel, normalize
from skimage import restoration
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr
import matplotlib.pyplot as plt
from registration import calculate_centroid
from scipy.ndimage import fourier_shift
from scipy.signal import convolve2d


def interpolation_restore(low_res, psf, downsample_factor, align_function=centroid_align):
    """
        Rudin's forward-projection method
        Reference: http://www.robots.ox.ac.uk/~vgg/publications/papers/capel01a.pdf (Sect. 5.8)

        Parameters
        ----------
        low_res : ndarray
            The low-resolution input images
        psf : ndarray
            The estimated point spread function used for deconvolution
        downsample_factor : integer
            Magnification factor to recover
        align_function : func
            The function used to align the low-resolution frames
    """
    # Align the stars according to their centroids
    low_res = align_function(low_res)

    # Average all of the low-resolution frames
    average_image = np.mean(low_res, axis=0)

    # Up-sample through bicubic interpolation
    high_res = misc.imresize(average_image, downsample_factor*100, interp='bicubic')

    # Deconvolve the PSF with standard single image method
    high_res = normalize(high_res, new_min=0, new_max=1)
    high_res = restoration.richardson_lucy(high_res, psf, iterations=15)

    return high_res


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


def mse(x, y):
    return np.linalg.norm(x-y)


def compare(original, restored):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    mse_original = mse(original, original)
    psnr_original = psnr(original, original)
    ssim_original = ssim(original, original)

    mse_restored = mse(original, restored)
    psnr_restored = psnr(original, restored)
    ssim_restored = ssim(original, restored)

    label = 'MSE: {:.2f}, PSNR: {:.2F}, SSIM: {:.2f}'

    ax[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_original, psnr_original, ssim_original))
    ax[0].set_title('Original image')

    ax[1].imshow(restored, cmap='gray', vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_restored, psnr_restored, ssim_restored))
    ax[1].set_title('Restored image')

    plt.tight_layout()
    plt.show()
