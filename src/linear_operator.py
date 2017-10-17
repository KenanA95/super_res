import numpy as np
import scipy.sparse as sparse


def construct_operator(images, M, N, downsample_factor, psf):

    operators = []

    for index in range(len(images)):
        dec_mat = decimation_matrix(M, N, downsample_factor)
        blur_mat = blur_matrix(M, N, psf)
        op = dec_mat * blur_mat

        operators.append(op)

    return sparse.vstack(operators, format='csr')


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


def decimation_matrix(M, N, downsample_factor):
    """
        Matrix operator to subsample an image by a given factor
        Reference: http://users.wfu.edu/plemmons/papers/siam_maa3.pdf ( sect. A.1)

        TODO: Fix to work on non-square images
        Parameters
        ----------
        M, N : int
            Dimensions of the high-resolution image
        downsample_factor : int
            How much to decimate the image by

        Returns
        -------
        D : (m**2, N**2) sparse array
            Sparse decimation matrix to down-sample an image through multiplication
            Where m is the size of the LR frame and N is the size of the HR image

    """
    m, n = int(M / downsample_factor), int(N / downsample_factor)

    # Get a grid the size of the lr frame to represent the indices
    rows, cols = np.meshgrid(range(m), range(n))

    # Change the sampling rate ex. (indices = 0, 1, 2..., factor = 2) => 0, 2, 4...
    sampled_rows = rows * downsample_factor
    sampled_cols = cols * downsample_factor

    # Convert the indices to be placed into the decimation matrix
    sparse_col_indices = np.ravel_multi_index((sampled_rows, sampled_cols), dims=(M, N)).T
    sparse_row_indices = np.ravel_multi_index((rows, cols), dims=(m, n)).T

    data = np.ones(len(sparse_row_indices) * m)

    return sparse.coo_matrix((data, (sparse_row_indices.flat, sparse_col_indices.flat)), shape=(m**2, N**2))


def out_of_bounds(mm, nn, M, N):
    return mm < 0 or mm >= M or nn < 0 or nn >= N


# TODO: Rewrite
def blur_matrix(M, N, psf):
    """
        Sparse block Toeplitz matrix (BTTB) to represent convolution through matrix multiplication
        Assumes zero boundary conditions and a spatially-invariant psf
        References: http://scholar.sun.ac.za/handle/10019.1/5189 ( sect. 7.2)
                    https://pdfs.semanticscholar.org/4c6c/d428cbcd75d257ef8de156cfbfd975bb7cfa.pdf
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
