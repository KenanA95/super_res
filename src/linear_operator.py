import numpy as np


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

    return X, Y