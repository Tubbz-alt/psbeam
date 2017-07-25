#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Low to mid level functions and classes that mostly involve contouring. For more
info on how they work, visit OpenCV's documentation on contours:

http://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html
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
from .images.templates import circle
from .preprocessing import threshold_image
from .beamexceptions import (NoContoursDetected, InputError)

logger = logging.getLogger(__name__)

def get_contours(image, thresh_mode="mean", *args, **kwargs):
    """
    Returns the contours of an image according to the inputted threshold.

    Parameters
    ----------
    image : np.ndarray
        Image to extract the contours from.

    thresh_mode : str, optional
    	Thresholding mode to use. For extended documentation see
    	preprocessing.threshold_image. Valid modes are:
    		['mean', 'top', 'bottom', 'adaptive', 'otsu']

    Returns
    -------
    contours : list
        A list of the contours found in the image.

    Raises
    ------
    NoContoursDetected
    	The returned contours list was empty.
    """
    image_thresh = threshold_image(image, mode=thresh_mode, **kwargs)
    _, contours, _ = cv2.findContours(image_thresh, 1, 2)    
    # Check if contours is empty
    if not contours:
        raise NoContoursDetected
    return contours

def get_largest_contour(image=None, contours=None, thresh_mode="mean",
                        **kwargs):
    """
    Returns largest contour of the contour list. Either an image or a contour
    must be passed. If both are passed, a warning will be logged and the
    contours of the image will be computed and used to find the largest contour.

    Function is making an implicit assumption that there will only be one
    (large) contour in the image. 

    Parameters
    ----------
    image : np.ndarray, optional
        Image to extract the contours from.

    contours : np.ndarray, optional
        Contours found on an image.

    thresh_mode : str, optional
    	Thresholding mode to use. For extended documentation see
    	preprocessing.threshold_image. Valid modes are:
    		['mean', 'top', 'bottom', 'adaptive', 'otsu']
    
    Returns
    -------
    np.ndarray
        Contour that encloses the largest area.
    """
    # Check if contours were inputted
    if image is None and contours is None:
        raise InputError("No image or contours provided.")
    elif image is not None:
        if contours is not None:
            logger.warn("Image and contours inputted. Returning largest contour"
                        " of the image.")
        contours = get_contours(image, thresh_mode=thresh_mode, **kwargs)
        
    # Get area of all the contours found
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    # Return argmax and max
    return contours[np.argmax(areas)], areas.max()

def get_moments(image=None, contour=None, **kwargs):
    """
    Returns the moments of an image.

    Attempts to find the moments using an inputted contours first, but if it
    isn't inputted it will compute the contours of the image then compute
    the moments.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate moments from.

    contour : np.ndarray
        Beam contour.

    Returns
    -------
    list
        List of zero, first and second image moments for x and y.
    """
    if image is None and contour is None:
        raise InputError("No image or contour provided.")
    elif image is not None:
        if contour is not None:
            logger.warn("Image and contour inputted. Using largest contour "
                        "of the image.")
        contour, _ = get_largest_contour(image, **kwargs)
        
    return cv2.moments(contour)
        
def get_centroid(M):
    """
    Returns the centroid using the inputted image moments.

    Centroid is computed as being the first moment in x and y divided by the
    zeroth moment.

    Parameters
    ----------
    M : list
        List of image moments.
    
    Returns
    -------
    tuple
        Centroid of the image moments.
    """    
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def get_bounding_box(image=None, contour=None, **kwargs):
    """
    Finds the up-right bounding box that contains the inputted contour. Either
    an image or contours have to be passed.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate moments from.

    contour : np.ndarray
        Beam contour.

    Returns
    -------
    tuple
        Contains x, y, width, height of bounding box.

    It should be noted that the x and y coordinates are for the bottom left
    corner of the bounding box. Use matplotlib.patches.Rectangle to plot.
    """
    if image is None and contours is None:
        raise InputError("No image or contour provided.")
    elif image is not None:
        if contours is not None:
            logger.warn("Image and contour inputted. Using largest contour "
                        "of the image.")
        contour, _ = get_largest_contour(image, **kwargs)
        
    return cv2.boundingRect(contour)

def get_contour_size(image=None, contour=None, **kwargs):
    """
    Returns the length and width of the contour, or the contour of the image
    inputted.

    Parameters
    ----------
    image : np.ndarray
        Image to calculate moments from.

    contour : np.ndarray
        Beam contour.

    Returns
    -------
    tuple
        Length and width of the inputted contour.
    """
    _, _, w, l = get_bounding_box(image=image, contour=contour, **kwargs)
    return l, w

# Define circle_contour as a global
_circle_contour, _ = get_largest_contour(circle, factor=0)
    
def get_circularity(contour, method=1):
    """
    Returns a score of how circular a contour is by comparing it to the
    contour of the template image, "circle.png." in the template_images
    directory.

    Parameters
    ----------
    contour : np.ndarray
        Contour to be compared with the circle contour

    method : int, optional
        Matches the contours according to an enumeration from 0 to 2. To see
        the methods in detail, go to:
        http://docs.opencv.org/3.1.0/df/d4e/group__imgproc__c.html#gacd971ae682604ff73cdb88645725968d

    Returns
    -------
    float
        Value ranging from 0.0 to 1.0 where 0.0 is perfectly similar to a 
        circle.
    """
    return cv2.matchShapes(_circle_contour, contour, method, 0)
