# super_res

Super-resolution is a set of algorithms designed to enhance image quality
by combining complementary information found throughout different low-resolution
frames.


For more information and examples see docs


###  Project Introduction
NASA’s OSIRIS-REx asteroid sample return mission spacecraft includes the
Touch And Go Camera System (TAGCAMS) three camera-head instrument. The
purpose of TAGCAMS is to provide imagery during the mission to
facilitate navigation to the target asteroid, confirm acquisition of
the asteroid sample and document asteroid sample stowage. Two of the
TAGCAMS cameras, NavCam1 and NavCam2, serve as fully redundant navigation
cameras to support optical navigation and natural feature tracking.

Images acquired by the NavCams typically contain star fields which provide
low-resolution representations of stars across the field of view. Multiple
exposures of the same scene are usually acquired with subpixel changes in
pointing between image frames.

*Synthetic Example*

![alt text](https://user-images.githubusercontent.com/15075964/32007516-c0d6da32-b977-11e7-881e-c1ca7f2e574e.png)



The stars geometric spatial extent are significantly smaller than the width of the
imaging system's pixels, and can therefore be considered a point source
representative of the camera's point spread function (PSF).The PSF is the
camera’s impulse response, and introduces blur into the image.


![alt text](https://user-images.githubusercontent.com/15075964/32007531-cc950682-b977-11e7-82e6-8c5f7901297d.png)


Super-resolution techniques allow us to generate a high-resolution representation
of the PSF by  integrating information found throughout several low-resolution images.
To identify the most effective method of constructing a higher-resolution image,
this project applied conventional SR algorithms alongside the emerging method
of convolutional neural networks.