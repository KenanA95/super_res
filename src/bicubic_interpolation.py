import numpy as np
from registration import centroid_align
from scipy import misc
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr, compare_mse as mse
import matplotlib.pyplot as plt
import warnings


# Super-resolution by interpolating the averaged image


def bicubic_restore(low_res, downsample_factor, align_function=centroid_align):
    """
        Rudin's forward-projection method
        Reference: http://www.robots.ox.ac.uk/~vgg/publications/papers/capel01a.pdf (Sect. 5.8)

        Parameters
        ----------
        low_res : ndarray
            The low-resolution input images
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

    return high_res


def normalize(im, new_min, new_max):
    """
        Linear normalization to convert image to any range of values

                        (new_max-new_min)
        In = (I-Min) * ___________________ + new_min
                           max - min

    """
    return (im-im.min()) * (new_max - new_min) / (im.max() - im.min()) + new_min


def compare(original, restored):
    """
        Side by side comparison of the original image and reconstruction effort with MSE, PSNR, and SSIM labels
    """
    if original.dtype != restored.dtype:
        warnings.warn("The images are different data types. Converting both images to floats within 0-1 range")
        original = normalize(original.astype(float), 0, 1)
        restored = normalize(restored.astype(float), 0, 1)

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
