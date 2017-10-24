import numpy as np
from scipy.ndimage.filters import gaussian_filter

# Find and grab the stars from the NavCam images. The stars are used as point sources to model the point
# spread function.


def get_grid(im, center, size):
    """Extract a sub-matrix from the image array around a given point """
    offset = int((size - 1) / 2)

    return im[center[0] - offset:center[0] + offset + 1,
              center[1] - offset:center[1] + offset + 1]


def set_grid(im, center, size, value=0):
    """ Set a sub-matrix within an image to a given value """
    offset = int((size - 1) / 2)

    im[center[0] - offset:center[0] + offset + 1,
       center[1] - offset:center[1] + offset + 1] = value

    return im


def star_locations(im, n, sigma, saturation_thresh):
    """
        Find the locations of stars in the NavCam images by blurring the image and selecting the brightest locations

        Parameters
        ----------
        im: ndarray
            Image to search in
        n: int
            The number of locations to find
        sigma
            Sigma value for the gaussian filter. Controls the level of blur when searching for stars
        saturation_thresh
            DN value indicating a pixel is over saturated and the star should be ignored

        Returns
        -------
        locations: list
            List of (x, y) pixel coordinates

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
        Extract the stars from a NavCam image by locating the brightest points

        Parameters
        ----------
        im: ndarray
            The image to grab the stars from
        locations: list
            List of the stars pixel coordinates
        size: int
            The diameter of the grid. Ex. size=5 -> grid=5x5

        Returns
        -------
        stars: list
            A list of the stars represented as matrices

    """
    stars = []
    for loc in locations:
        star = get_grid(im, loc, size)
        # Necessary because stars on the edge of the image are cut off and can ruin later analysis
        if star.shape == (size, size):
            stars.append(star)

    return stars


def divide_image(image, shape):
    """
        Split an image into a list of blocks
    """
    x, y = shape
    vertical_sections = np.vsplit(image, x)
    blocks = np.empty((x * y), dtype=object)

    for row, section in enumerate(vertical_sections):
        sub_sections = np.array(np.hsplit(section, y)).astype(float)

        for col, block in enumerate(sub_sections):
            index = np.ravel_multi_index((row, col), shape)
            blocks[index] = block

    return blocks


def extract_stars_by_section(images, shape, n, size, sigma=2, saturation_thresh=3500):
    """

        Divides each image into sections and extracts the stars of each section across all images

        Parameters
        ----------
        images: list
            The NavCam images to extract the stars from
        shape: tuple
            Shape to divide each image into
        n: int
            Number of stars to extract from each section
        size: int
            Size of the extracted star. i.e size=7 => star = 7x7
        sigma
            Sigma value for the gaussian filter. Controls the level of blur when searching for stars
        saturation_thresh
            DN value indicating a pixel is over saturated and the star should be ignored

        Returns
        -------
        section_stars: dict
            The extracted stars of every section in the form of a dictionary. The key to any section is
            its numeric index

    """
    # Divide each image into a list of sections
    divided_images = [divide_image(im, shape) for im in images]

    # Dictionary to store the stars from each section across all images
    keys = np.arange(np.prod(shape))
    section_stars = {key: [] for key in keys}

    for im in divided_images:
        for i, section in enumerate(im):
            star_locs = star_locations(section, n, sigma, saturation_thresh)
            stars = extract_stars(section, star_locs, size)
            section_stars[i].extend(stars)

    return section_stars
