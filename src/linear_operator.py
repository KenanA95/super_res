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


# Matrix to decimate a single high-res image by a given factor
def decimation_matrix(target_resolution, downsample_factor):

    lr_size = int(target_resolution / downsample_factor)

    # Get a grid the size of the lr frame to represent the indices
    row_indices, col_indices = np.meshgrid(range(lr_size), range(lr_size))

    # Change the sampling rate ex. (indices = 0, 1, 2..., factor = 2) => 0, 2, 4...
    row_indices = row_indices * downsample_factor
    col_indices = col_indices * downsample_factor

    # Convert the indices to the decimation matrix
    sparse_col_indices = np.ravel_multi_index((row_indices, col_indices), dims=(target_resolution, target_resolution)).T
    sparse_row_indices = np.ravel_multi_index((lr_size, lr_size), dims=(lr_size, lr_size)).T

    data = np.ones(len(sparse_row_indices) * lr_size)
    shape = (lr_size ** 2, target_resolution ** 2)

    return sparse.coo_matrix((data, (sparse_row_indices.flat, sparse_col_indices.flat)), shape)
