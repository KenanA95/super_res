import numpy as np
from scipy.sparse.linalg import lsqr
from registration import centroid_align
from bicubic_interpolation import bicubic_restore
from linear_operator import construct_operator
from scipy.signal import convolve2d


# Stack all the low-resolution images into the vector b in lexicographical order
def stack_low_res(low_res):
    lr_size = np.prod(low_res[0].shape)
    b = np.empty(len(low_res) * lr_size)

    for i in range(len(low_res)):
        b[i * lr_size:(i + 1) * lr_size] = low_res[i].flat

    return b


def gradient_descent(low_res, A, x0, iterations, damp=1e-1):
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
    M, N = x0.shape[0], x0.shape[1]

    x = x0.copy().flat

    # Steepest descent
    for i in range(iterations):
        step = damp * -1 * (A.T * ((A * x) - b))
        prior = damp * np.subtract(x, x0.flat)
        x += step  # + prior

    return np.reshape(x, (M, N))


def lsqr_restore(low_res, A, x0, iter_lim, atol=1e-8, btol=1e-8, damp=1e-1):
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
    M, N = x0.shape[0], x0.shape[1]

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


def irani_peleg_restore(low_res, psf, downsample_factor, iterations, k=1, align_function=centroid_align):
    """
        Irani-Peleg iterative back-projection method. Very much a work in progress
        Reference: http://www.cse.huji.ac.il/course/2003/impr/supres-cvgip-gm91.pdf

        Parameters
        ----------
        low_res: list
            A list of the low_resolution input stars that are already aligned centrally
        psf: ndarray
            Matrix representation of the point spread function
        downsample_factor: int
            Magnification factor to recover
        iterations: int
        k: int
        align_function: func
            The function used to align the stars when creating the initial avaeraged image

        Returns
        -------
        x: ndarray
            The high-resolution estimate

    """
    # Initial estimate is the upsampled averaged image
    x0 = bicubic_restore(low_res, downsample_factor, align_function)

    # Dimensions of the high-resolution image
    M, N = x0.shape[0], x0.shape[1]

    # Construct the operator that decimates and blurs a single image
    A = construct_operator(1, M, N, downsample_factor, psf)

    x = x0.copy()

    for i in range(iterations):
        error = 0

        for lr in low_res:
            simulated_lr = A * x.flat
            simulated_lr = np.reshape(simulated_lr, lr.shape)

            delta = simulated_lr - lr
            error += np.sum(abs(delta))

            delta = upsample_image(delta, downsample_factor)
            delta = convolve2d(delta, psf**k, 'same')

            x += delta

        print("Iteration: {0} \t Error: {1}".format(i, error))

    return x
