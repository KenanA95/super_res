import numpy as np
import scipy.sparse as sparse


def transform_coordinates(x, y, tf):
    """
        tf =
            [[cos(theta),     -sin(theta),  tx]
            [ sin(theta),     cos(theta),   ty]
            [0,               0,            1]]

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1
    """
    X = (tf[0, 0] * x + tf[0, 1] * y + tf[0, 2])
    Y = (tf[1, 0] * x + tf[1, 1] * y + tf[1, 2])

    return int(X), int(Y)


def decimation_matrix(target_resolution, downsample_factor):
    """
        Matrix operator to subsample an image by a given factor
        Reference: http://users.wfu.edu/plemmons/papers/siam_maa3.pdf ( sect. A.1)

        Parameters
        ----------
        target_resolution : int
            Size of the high-resolution image
        downsample_factor : int
            How much to decimate the image by

        Returns
        -------
        D : (M**2, N**2) sparse array
            Sparse decimation matrix to down-sample an image through multiplication
            Where M is the size of the LR frame and N is the size of the HR image

    """
    lr_size = int(target_resolution / downsample_factor)

    # Get a grid the size of the lr frame to represent the indices
    rows, cols = np.meshgrid(range(lr_size), range(lr_size))

    # Change the sampling rate ex. (indices = 0, 1, 2..., factor = 2) => 0, 2, 4...
    sampled_rows = rows * downsample_factor
    sampled_cols = cols * downsample_factor

    # Convert the indices to the decimation matrix
    sparse_col_indices = np.ravel_multi_index((sampled_rows, sampled_cols),
                                              dims=(target_resolution, target_resolution)).T

    sparse_row_indices = np.ravel_multi_index((rows, cols), dims=(lr_size, lr_size)).T

    data = np.ones(len(sparse_row_indices) * lr_size)
    shape = (lr_size ** 2, target_resolution ** 2)

    return sparse.coo_matrix((data, (sparse_row_indices.flat, sparse_col_indices.flat)), shape)


def out_of_bounds(mm, nn, M, N):
    return mm < 0 or mm >= M or nn < 0 or nn >= N


# TODO: Rewrite
def blur_matrix(M, N, psf):
    """
        Sparse block Toeplitz matrix to represent convolution through matrix multiplication
        Assumes zero boundary conditions and a spatially-invariant psf
        Reference: http://scholar.sun.ac.za/handle/10019.1/5189 ( sect. 7.2)

        Parameters
        ----------
        M, N : int
            Size of the high-resolution image

        psf: ndarray

        Returns
        -------
        H : (M*N, M*N) sparse array
            Sparse Toeplitz matrix with each row of the psf represented as a diagonal

    """
    row, col, data = [], [], []

    offset_diags = int((psf.shape[0] - 1) / 2)

    for r in range(M):
        for c in range(N):

            for i in range(-offset_diags, offset_diags + 1):
                for j in range(-offset_diags, offset_diags + 1):

                    mm = r + i
                    nn = c + j

                    if out_of_bounds(mm, nn, M, N):
                        continue

                    row.append(r * N + c)
                    col.append(mm * N + nn)
                    data.append(psf[i + offset_diags, j + offset_diags])

    return sparse.coo_matrix((data, (row, col)), shape=(M*N, M*N)).tocsr()