from scipy import ndimage
from skimage.filters import threshold_otsu


def calculate_centroid(im):
    thresh = threshold_otsu(im)
    binary = im > thresh
    return ndimage.measurements.center_of_mass(binary)
