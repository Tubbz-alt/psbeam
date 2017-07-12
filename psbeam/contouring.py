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
from .template_images import circle_small
from .beamexceptions import NoContoursPresent

logger = logging.getLogger(__name__)

def get_contours(image, factor=3):
    """
    Returns the contours of an image according to a mean-threshold.

    Parameters
    ----------
    image : np.ndarray
    	Image to extract the contours from.

    factor : int, optional
    	Number of times to multiply the std by before adding to the mean for
    	thresholding.
    """
    _, image_thresh = cv2.threshold(
        image, image.mean() + image.std()*factor, 255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(image_thresh, 1, 2)
    return contours

def get_largest_contour(image=None, contours=None, factor=3):
    """
    Returns largest contour of the contour list. Either an image or a contour
    must be passed.

    Function is making an implicit assumption that there will only be one
    (large) contour in the image. 

    Parameters
    ----------
    image : np.ndarray, optional
    	Image to extract the contours from.

    contours : np.ndarray, optional
    	Contours found on an image.

    factor : int, optional
    	Number of times to multiply the std by before adding to the mean for
    	thresholding.    

    Returns
    -------
    np.ndarray
    	Contour that encloses the largest area.
    """
    # Check if contours were inputted
    if contours is None:
        contours = get_contours(image, factor=factor)
    # Check if contours is empty
    if not contours:
        raise NoContoursPresent
    # Get area of all the contours found
    area = [cv2.contourArea(cnt) for cnt in contours]
    # Return argmax and max
    return contours[np.argmax(np.array(area))], np.array(area).max()

def get_moments(image=None, contour=None):
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
    try:
        return cv2.moments(contour)
    except TypeError:
        contour, _ = get_largest_contour(image)
        return cv2.moments(contour)

def get_centroid(M):
    """
    Returns the centroid using the inputted image moments.

    Centroid is computed as being the first moment in x and y divided by the
    zeroth moment.

    Parameters
    ----------
    M : list
    	Moments of an image
    
    Returns
    -------
    tuple
    	Centroid of the image
    """    
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def get_bounding_box(image=None, contour=None):
    """
    Finds the up-right bounding box that contains the inputted contour. Either
    an image or contours have to be passed.

    Parameters
    ----------
    image : np.ndarray, optional
    	Image to get a bounding box for.

    contour : np.ndarray, optional
    	Beam contour.

    Returns
    -------
    tuple
    	Contains x, y, width, height of bounding box.

    It should be noted that the x and y coordinates are for the bottom left
    corner of the bounding box. Use matplotlib.patches.Rectangle to plot.
    """
    try:
        return cv2.boundingRect(contour)
    except TypeError:
        contour, _ = get_largest_contour(image)
        return cv2.boundingRect(contour)

# Define circle_small_contour as a global
circle_small_contour, _ = get_largest_contour(circle_small, factor=0)
    
def get_circularity(contour, method=1):
    """
    Returns a score of how circular a contour is by comparing it to the
    contour of the template image, "circle_small.png." in the template_images
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
    	Value ranging from 0.0 to 1.0 where 0.0 is perfectly similar to a circle
    """
    return cv2.matchshapes(circle_small_contour, contour, method, 0)
