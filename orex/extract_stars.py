import numpy as np
from scipy.ndimage.filters import gaussian_filter

# Find and grab the stars from the NavCam images. The stars are used as point sources to model the point
# spread function.


def get_grid(im, center, size):
    """Extract a sub-matrix from the image array around a given point

    :param im: The image to extract the sub-matrix from
    :param center: The center of the grid to be extracted
    :param size: The diameter of the grid. Ex. size=5 -> grid=5x5
    :return: A sub-matrix from the image around the center coordinates provided
    """
    offset = int((size - 1) / 2)

    return im[center[0] - offset:center[0] + offset + 1,
              center[1] - offset:center[1] + offset + 1]


def set_grid(im, center, size, value=0):
    """ Set a sub-matrix within an image to a given value

    :param im: The image the sub-matrix is in
    :param center: The center of the grid to be set
    :param size: The diameter of the grid. Ex. size=5 -> grid=5x5
    :param value: The value to set the sub-matrix to
    :return: The altered image
    """
    offset = int((size - 1) / 2)

    im[center[0] - offset:center[0] + offset + 1,
       center[1] - offset:center[1] + offset + 1] = value

    return im


def star_locations(im, n, sigma, saturation_thresh):
    """ Find the locations of stars in the NavCam images by blurring the image and selecting the brightest locations

    :param im: The image to search in
    :param n: The number of locations to fine
    :param sigma: The sigma value for the gaussian filter. Controls the level of blur
    :param saturation_thresh: DN value indicating a pixel is over saturated and the star data is not reliable
    :return: x, y pixel coordinates of the stars stored in a list
    """
    height, width = im.shape[:2]
    blurred = gaussian_filter(im, sigma=sigma)
    locations = []

    while n != 0:
        # Location of the brightest pixel in the blurred image
        max_location = np.unravel_index(np.argmax(blurred), (height, width))

        # Used to check if the star is over saturated
        star = get_grid(im, max_location, 5)

        # Remove the brightest location so the next time it finds a different location
        blurred = set_grid(blurred, max_location, size=5, value=0)

        # If the star contains a value that is saturated ignore it and move on
        if len(np.where(star >= saturation_thresh)[0]):
            continue

        locations.append(max_location)
        n -= 1

    return locations


def extract_stars(im, locations, size):
    """

    :param im: The image to grab the stars from
    :param locations: List of the stars pixel coordinates
    :param size: The diameter of the grid. Ex. size=5 -> grid=5x5
    :return: A list of the stars represented as matrices
    """
    stars = []
    for loc in locations:
        star = get_grid(im, loc, size)
        # Necessary because stars on the edge of the image are cut off and can ruin later analysis
        if star.shape == (size, size):
            stars.append(star)

    return stars


# Split an image into a list of blocks
def divide_image(image, shape):
    x, y = shape
    vertical_sections = np.vsplit(image, x)
    blocks = np.empty((x * y), dtype=object)

    for row, section in enumerate(vertical_sections):
        sub_sections = np.array(np.hsplit(section, y)).astype(float)

        for col, block in enumerate(sub_sections):
            index = np.ravel_multi_index((row, col), shape)
            blocks[index] = block

    return blocks
