import numpy as np
from scipy.sparse.linalg import lsqr
from registration import centroid_align
from scipy import misc
from observation_model import ObservationModel, normalize
from registration import calculate_centroid
from scipy.ndimage import fourier_shift
from scipy.signal import convolve2d


# Stack all the low-resolution images into the vector b in lexicographical order
def stack_low_res(low_res):
    lr_size = np.prod(low_res[0].shape)
    b = np.empty(len(low_res) * lr_size)

    for i in range(len(low_res)):
        b[i * lr_size:(i + 1) * lr_size] = low_res[i].flat

    return b


def gradient_descent(low_res, A, x0, upsample_factor, iterations, damp=1e-1):
    """
        Solve Ax=b through steepest descent optimization
        Reference: http://scholar.sun.ac.za/handle/10019.1/5189 ( sect. 7.4.1)

        Parameters
        ----------
        low_res: list
            A list of the low_resolution input frames
        A: Sparse CSR matrix
            Sparse operator representing Decimation + Blur
        x0: ndarray
            The initial guess for the high-resolution image. Typically the upsampled average image
        upsample_factor: int
            How much to scale the resolution
        iterations: int
            Number of iterations performed for the steepest descent
        damp: float
            The weight given to the initial guess. The larger the value, the closer the result is to x0

        Returns
        -------
        x: ndarray
            The high-resolution estimate
    """

    # Stack all the low-resolution images into the vector b in lexicographical order
    b = stack_low_res(low_res)

    # Get the dimensions of the new high-resolution image
    M, N = low_res[0].shape[0] * upsample_factor, low_res[0].shape[1] * upsample_factor

    x = x0.copy().flat

    # Steepest descent
    for i in range(iterations):
        step = damp * -1 * (A.T * ((A * x) - b))
        prior = damp * np.subtract(x, x0.flat)
        x += step  # + prior

    return np.reshape(x, (M, N))


def lsqr_restore(low_res, A, x0, upsample_factor, iter_lim, atol=1e-8, btol=1e-8, damp=1e-1):
    """
            Solve Ax=b through least-solutions solution
            Reference: http://scholar.sun.ac.za/handle/10019.1/5189 ( sect. 7.4.1)

            Parameters
            ----------
            low_res: list
                A list of the low_resolution input frames
            A: Sparse CSR matrix
                Sparse operator representing Decimation + Blur
            x0: ndarray
                The initial guess for the high-resolution image. Typically the upsampled average image
            upsample_factor: int
                How much to scale the resolution
            iter_lim: int
               Explicit limitation on number of iterations (for safety)
            damp: float
                Damping coefficient

            Returns
            -------
            x: ndarray
                The high-resolution estimate
    """
    # Stack all the low-resolution images into the vector b in lexicographical order
    b = stack_low_res(low_res)

    # Get the dimensions of the new high-resolution image
    M, N = low_res[0].shape[0] * upsample_factor, low_res[0].shape[1] * upsample_factor

    x0 = x0.flat
    b = b - A*x0

    x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = \
        lsqr(A, b, damp, atol, btol, iter_lim=iter_lim, show=True)

    x = x0 + x

    return np.reshape(x, (M, N))


def upsample_image(im, n):
    """
         Upsample the image to n x the original size of each dimension,
         with the new even rows and columns filled with zeros (0)

    """
    new_shape = np.multiply(im.shape, n)
    upsampled_im = np.zeros(new_shape)
    upsampled_im[0::n, 0::n] = im

    return upsampled_im


# Irani-Peleg iterative back-projection method. Very much a work in progress
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