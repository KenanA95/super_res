import numpy as np
from scipy.sparse.linalg import lsqr


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
