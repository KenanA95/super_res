

class LowResolution:

    def __init__(self, image, transform_matrix, downsample_factor):
        self.image = image
        self.transform_matrix = transform_matrix
        self.downsample = downsample_factor
