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

def to_uint8(image, mode="norm"):
    """
    *Correctly* converts an image to uint8 type.
    
    Running 'image.astype(np.uint8)' on its own applies a mod(256) to handle
    values over 256. The correct way is to either clip (implemented here) or
    normalize.

    Parameters
    ----------
    image : np.ndarray
        Image to be converted to uint8.

    mode : str, optional 
        Conversion mode that either clips the image or normalizes to the range
        of the original image type.

    Returns
    -------
    np.ndarray
        Image that is an np.ndarray of dtype uint8.
    """
    # Make sure the image is a numpy array
    if not isinstance(image, np.ndarray):
        image_array = np.array(image)
    else:
        image_array = np.copy(image)
    # Clip or normalize the image
    if mode.lower() == "clip":
        return cv2.convertScaleAbs(image_array)
    elif mode.lower() == "norm":
        # Normalize according to the array max and min values
        type_min = np.iinfo(image_array.dtype).min
        type_max = np.iinfo(image_array.dtype).max
        return cv2.convertScaleAbs(image_array, alpha=255/(type_max-type_min))
    # Handle invalid inputs
    raise InputError("Invalid conversion mode inputted. Valid modes are " \
                     "'clip' and 'norm.'")

def uint_resize_gauss(image, mode='norm', fx=1.0, fy=1.0, kernel=(11,11), 
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

    mean:
        Set the threshold line to be image.mean + image.std*factor.
    top:
        Sets the threshold line to be image.max - image.std*factor, leaving just
        the highest intensity pixels.
    bottom:
        Sets the threshold line to be image.min + image.std*factor, removing
        just the lowest intensity pixels
    adaptive:
        Sets threshold line according to a weighed sum of neughborhood values
        using a gaussian window. See 'Adaptive Thresholding' in the following
        link for more details.
        http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    otsu:
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
        _, th = cv2.threshold(image, image.max() + image.std()*factor, 255,
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
        
                    
