from linear_operator import construct_operator
import numpy as np


def gradient_descent(low_res, psf, x0, upsample_factor, iterations, damp=1e-1):
    """
        Solve Ax=b through steepest descent optimization
        Reference: http://scholar.sun.ac.za/handle/10019.1/5189 ( sect. 7.4.1)

        Parameters
        ----------
        low_res: list
            A list of the low_resolution input frames
        psf: ndarray
            An estimate of the Point Spread Function blurring the image
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
    lr_size = np.prod(low_res[0].shape)
    b = np.empty(len(low_res) * lr_size)

    for i in range(len(low_res)):
        b[i * lr_size:(i + 1) * lr_size] = low_res[i].flat

    # Get the dimensions of the new high-resolution image
    M, N = low_res[0].shape[0] * upsample_factor, low_res[0].shape[1] * upsample_factor

    # Linear operator representing blur and decimation
    A = construct_operator(len(low_res), M, N, downsample_factor=upsample_factor, psf=psf)

    x = x0.copy().flat

    # Steepest descent
    for i in range(iterations):
        step = damp * -1 * (A.T * ((A * x) - b))
        prior = damp * np.subtract(x, x0.flat)
        x += step  # + prior
        print("Gradient descent step {0}".format(i))

    return np.reshape(x, (M, N))
