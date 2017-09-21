import numpy as np
import math


class ObservationModel:

    def __init__(self, image, psf, downsample_factor, translation_range, rotation_range):
        self.image = image
        self.psf = psf
        self.downsample = downsample_factor
        self.translation_range = translation_range
        self.rotation_range = rotation_range
        self.low_resolution = []


    def transformation_matrix(self):
        """
        Generate a transformation matrix with random translation/rotation within the provided range

        :param translation_range: Min and max value in pixels that the random translation can be
        :param rotation_range: Min and max value that the random rotation angle can be (counter-clockwise)
        :return:
            [[cos(theta),     -sin(theta),  tx]
            [ sin(theta),     cos(theta),   ty]
            [0,               0,            1]]

        """

        theta = np.random.randint(self.rotation_range[0], self.rotation_range[1])
        tx, ty = np.random.randint(self.translation_range[0], self.translation_range[1], size=(1, 2))[0]

        return [
            [math.cos(theta), -math.sin(theta), tx],
            [math.sin(theta), math.cos(theta),  ty],
            [0, 0, 1]
        ]
