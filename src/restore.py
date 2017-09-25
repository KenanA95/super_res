import numpy as np
from registration import centroid_align
from scipy import misc
from observation_model import normalize
from skimage import restoration
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr
import matplotlib.pyplot as plt


def interpolation_restore(low_res, psf, downsample_factor, align_function=centroid_align,):

    # Align the stars according to their centroids
    low_res = align_function(low_res)

    # Average all of the low-resolution frames
    average_image = np.mean(low_res, axis=0)

    # Up-sample through bicubic interpolation
    high_res = misc.imresize(average_image, downsample_factor*100, interp='bicubic')

    # Deconvolve the PSF
    high_res = normalize(high_res, new_min=0, new_max=1)
    high_res = restoration.richardson_lucy(high_res, psf, iterations=15)

    return high_res


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

    ax[0].imshow(original, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_original, psnr_original, ssim_original))
    ax[0].set_title('Original image')

    ax[1].imshow(restored, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_restored, psnr_restored, ssim_restored))
    ax[1].set_title('Restored image')

    plt.tight_layout()
    plt.show()
