#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing functions used in psbeam.
"""
############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2
import numpy as np

##########
# Module #
##########
from .beamexceptions import InputError

logger = logging.getLogger(__name__)

def to_gray(image, color_space="RGB", cv2_color=None):
    """
    Converts the inputted image to gray scale.

    This function should serve as a wrapper for cv2.cvtColor with basic input
    checking. Its main use is to convert color images to gray scale but it can
    be used as an alias to cv2.cvtColor if a color conversion code is passed
    into cv2_color

    Parameters
    ----------
    image : np.ndarray
    	Image to convert to grayscale.

    color_space : str, optional
    	Color space of the image. Valid entries are 'RGB', 'BGR'.

    cv2_color : cv2.ColorConversionCodes
    	OpenCV color conversion code. Bypasses image array checks if used.

    Returns
    -------
    image_gray : np.ndarray
    	The image converted to gray scale (ie len(image.shape) is 2)

    Raises
    ------
    InputError
    	If input is not an image, image is not 3 channel or color_space is
    	invalid.
    """
    # Check if an OpenCV color conversion was entered
    if cv2_color is not None:
        color = cv2_color
        
    else:
        # Check we are getting an image
        if len(image.shape) < 2:
            raise InputError("Got array that is not an image. Shape is {0}."
                             "".format(image.shape))

        # Check that it isn't already grayscale
        if len(image.shape) < 3:
            raise InputError("Got image that is already grayscale.")

        # Check for the color space
        if color_space.upper() == "RGB":
            color = cv2.COLOR_RGB2GRAY
        elif color_space.upper() == "BGR":
            color = cv2.COLOR_BGR2GRAY
        else:
            raise InputError("Invalid color_space entry. Got '{0}'".format(
                color_space))

    # Do the conversion
    return cv2.cvtColor(image, color)

def to_uint8(image, mode="scale"):
    """
    *Correctly* converts an image to uint8 type.
    
    Running 'image.astype(np.uint8)' on its own applies a mod(256) to handle
    values over 256. The correct way is to either clip (implemented here) or
    normalize to the the max and min possible values of the array.

    Conversion Modes
    ----------------
    clip
    	Truncates the  image at 0 and 255, then returns the resulting array.

    norm
    	Normalizes the image so that the maximum value of the input array is set
    	to 255 and the minimum value is 0.
    
    scale
    	Scales the image so that the maximum and minimum values of the resulting
    	array corresponds to the maximum and minimum possible values of the
    	input array.

    Parameters
    ----------
    image : np.ndarray
        Image to be converted to uint8.

    mode : str, optional 
        Conversion mode to use. See conversion modes for more details.

    Returns
    -------
    np.ndarray
        Image that is a np.ndarray with dtype uint8.
    """
    # The main hurdle with performing the clipping is making sure we don't
    # create values that exceed the buffer size of the inputted dtype. For
    # example, for float32 the maximum value is 65504.0 and the min value is
    # 65504.0. If you try to find the range using a float32 by doing 65504.0 -
    # (-65504.0) you will get Inf.
    
    # Make sure the image is a numpy array
    if not isinstance(image, np.ndarray):
        image_array = np.array(image)
    else:
        image_array = np.copy(image)
        
    # Clip or normalize the image
    if mode.lower() == "clip":
        output = np.clip(image_array, 0, 255)

    # Normalize to max and min values of the array
    elif mode.lower() == "norm":
        range_pixels = image_array.max() - image_array.min() or 1        
        output = 255 * (image_array - image_array.min()) / range_pixels
        
    # Scale according to the max and min possible values of the array
    elif mode.lower() == "scale":
        try:
            # Grab array info for int types
            type_min = np.iinfo(image_array.dtype).min
            type_max = np.iinfo(image_array.dtype).max
        except ValueError:
            # Grab array info for float types
            type_min = np.finfo(image_array.dtype).min
            type_max = np.finfo(image_array.dtype).max
        # Use half the range to guarantee we can fit the range in the variable
        range_half = type_max/2 - type_min/2
        output = ((image_array/2 - type_min/2)/range_half) * 255

    # Handle invalid inputs
    else:
        raise InputError("Invalid conversion mode inputted. Valid modes are " 
                         "'clip' and 'norm.'")

    # Warn the user if the preprocessing resulted in a zeroed array
    if not output.any() and image_array.any():
        logger.warn("to_uint8 resulted in a fully zeroed array from non-zero "
                    "input.")

    # Return casting as uint8
    return output.astype(np.uint8)

def uint_resize_gauss(image, mode='scale', fx=1.0, fy=1.0, kernel=(11,11), 
                      sigma=0):
    """
    Preprocess the image by converting to uint8, resizing and running a 
    gaussian blur.
    
    Parameters
    ----------
    image : np.ndarray
        The image to be preprocessed.

    mode : str, optional 
        Conversion mode that either clips the image or normalizes to the range
        of the original image type.

    fx : float, optional
        Percent to resize the image by in x.

    fy : float, optional
        Percent to resize the image by in y.

    kernel : tuple, optional
        Kernel to use when running the gaussian filter.

    Returns
    -------
    np.ndarray 
        Image of type uint8 that has been resized and blurred.
    """
    image_uint = to_uint8(image, mode=mode)
    image_resized = cv2.resize(image_uint, (0,0), fx=fx, fy=fy)
    image_gblur = cv2.GaussianBlur(image_resized, kernel, sigma)
    return image_gblur

def threshold_image(image, binary=True, mode="top", factor=3, **kwargs):
    """
    Thresholds the image according to one of the modes described below.

    Threshold Modes
    ---------------
    mean
        Set the threshold line to be image.mean + image.std*factor.

    top
        Sets the threshold line to be image.max - image.std*factor, leaving just
        the highest intensity pixels.

    bottom
        Sets the threshold line to be image.min + image.std*factor, removing
        just the lowest intensity pixels

    adaptive
        Sets threshold line according to a weighed sum of neughborhood values
        using a gaussian window. See 'Adaptive Thresholding' in the following
        link for more details.
        http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html

    otsu
        Sets the threshold to be between the histogram peaks of a bimodal image.
        See "Otsu's Binarization" in the following for more details.
        http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html

    Parameters
    ----------
    image : np.ndarray
        The image to threshold.

    binary : bool, optional
        Use binary thresholding or to_zero thresholding.

    mode : str, optional
        Thresholding mode to use. See docstring for more information.

    factor : int, optional
        Number of times to multiply the std by before adding to the mean for
        thresholding.    
        
    Returns
    -------
    th : np.ndarray
        Image that has been thresholded.
    """
    valid_modes = set('mean', 'top', 'bottom', 'adaptive', 'otsu')
    if binary:
        th_type = cv2.THRESH_BINARY
    else:
        th_type = cv2.THRESH_TOZERO

    # Find the right mode
    if mode.lower() not in valid_modes:
        raise InputError("Invalid mode passed for thresholding.")
    elif mode.lower() == 'mean':
        _, th = cv2.threshold(image, image.mean() + image.std()*factor, 255,
                              th_type)        
    elif mode.lower() == 'top':
        _, th = cv2.threshold(image, image.max() - image.std()*factor, 255,
                              th_type)
    elif mode.lower() == 'bottom':
        _, th = cv2.threshold(image, image.min() + image.std()*factor, 255,
                              th_type)
    elif mode.lower() == "adaptive":
        blockSize = kwargs.pop("BlockSize", 11)
        C = kwargs.pop("C", 2)        
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   th_type, blockSize, C)
    elif mode.lower() == "otsu":
        _, th = cv2.threshold(image, 0, 255, th_type+cv2.THRESH_OTSU)

    return th
        
                    
