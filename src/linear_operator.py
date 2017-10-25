import numpy as np
import scipy.sparse as sparse
from itertools import product

__doc__ = """

Construct an operator to solve Ax=b where
     A is the sparse operator representing Decimation + Homogeneous Transformation
     x is the high-resolution target
     b is a stacked vector of all the low-resolution images

    The blur matrix is not included because we are working with the special case where
    the high-resolution target is the PSF. I included an example of how it can be built for other cases.
"""


def construct_operator(im_count, M, N, downsample_factor, psf):
    """
        Linear operator to represent the image observation model

        Parameters
        ----------
        im_count: int
            Number of images
        M, N: int
            Dimensions of the high-resolution image
        downsample_factor: int
            How much to decimate the images by
        psf: ndarray
            Matrix representation of the point spread function
        Returns
        -------
        operator: ndarray with dimensions (im_count * m*n, M*N) where m,n are the dimensions of a low-resolution frame
            Sparse linear operator representing the image observation model

    """

    dec_mat = decimation_matrix(M, N, downsample_factor)
    blur_mat = blur_matrix(M, N, psf)
    operator = dec_mat * blur_mat
    operator = np.repeat(operator, im_count, axis=0)

    return sparse.vstack(operator, format='csr')


def decimation_matrix(M, N, downsample_factor):
    """
        Matrix operator to subsample an image by a given factor
        Reference: http://users.wfu.edu/plemmons/papers/siam_maa3.pdf ( sect. A.1)

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


def transformation_matrix(tf, image_shape):
    """
        Represent homographic transformation as linear operator using bilinear interpolation
        Source: https://github.com/elegant-scipy/elegant-scipy/blob/master/markdown/ch5.markdown

        Parameters
        ----------
        tf : (3, 3) ndarray
            Transformation matrix.
        image_shape : (M, N)
            Shape of input  image

        Returns
        -------
        A : (M * N, M * N) sparse matrix
            Linear-operator representing transformation + bilinear interpolation.

    """
    # Invert matrix.  This tells us, for each output pixel, where to find its corresponding input pixel.
    H = np.linalg.inv(tf)

    m, n = image_shape

    row, col, values = [], [], []

    # For each pixel in the output image...
    for sparse_op_row, (out_row, out_col) in \
            enumerate(product(range(m), range(n))):

        # Compute where it came from in the input image
        in_row, in_col, in_abs = H @ [out_row, out_col, 1]
        in_row /= in_abs
        in_col /= in_abs

        # if the coordinates are outside of the original image, we will have 0 at this position
        if (not 0 <= in_row < m - 1 or
                not 0 <= in_col < n - 1):
            continue

        # Use the four surrounding pixels to interpolate the output pixel value
        top = int(np.floor(in_row))
        left = int(np.floor(in_col))

        # Calculate the position of the output pixel, mapped into the input image, within the four selected pixels
        t = in_row - top
        u = in_col - left

        # The current row of the sparse operator matrix is given by the raveled output pixel coordinates,
        # contained in `sparse_op_row`. Four surrounding input pixels correspond to four columns so repeat four times
        row.extend([sparse_op_row] * 4)

        # Weighted values are calculated according to the bilinear interpolation algorithm
        sparse_op_col = np.ravel_multi_index(
                ([top,  top,      top + 1, top + 1],
                 [left, left + 1, left,    left + 1]), dims=(m, n))
        col.extend(sparse_op_col)
        values.extend([(1-t) * (1-u), (1-t) * u, t * (1-u), t * u])

    return sparse.coo_matrix((values, (row, col)), shape=(m*n, m*n)).tocsr()


def out_of_bounds(mm, nn, M, N):
    return mm < 0 or mm >= M or nn < 0 or nn >= N


# TODO: Rewrite blur operator
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
