# super_res

For more information please see docs

## Introduction
NASA’s OSIRIS-REx asteroid sample return mission spacecraft includes the
Touch And Go Camera System (TAGCAMS) three camera-head instrument. The
purpose of TAGCAMS is to provide imagery during the mission to
facilitate navigation to the target asteroid, confirm acquisition of
the asteroid sample and document asteroid sample stowage. Two of the
TAGCAMS cameras, NavCam1 and NavCam2, serve as fully redundant navigation
cameras to support optical navigation and natural feature tracking. Images
acquired by the NavCams typically contain star fields which provide
low-resolution representations of the camera’s point spread function
(PSF) across the field of view. The PSF is the camera’s impulse response,
or how the camera represents a point source smaller than the maximum
resolution of a pixel and introduces blur into the image. In an image,
an object can be considered a point source if its geometric spatial extent
is significantly smaller than the width of the imaging system’s pixels.
Multiple exposures of the same scene are usually acquired with subpixel
changes in pointing between image frames. Super-resolution (SR) techniques
allow us to generate a high-resolution (HR) representation of the PSF by
integrating information found throughout several low-resolution (LR) images.
To identify the most effective method of constructing a higher-resolution
NavCam image, this project applied conventional SR algorithms alongside
the emerging method of convolutional neural networks.