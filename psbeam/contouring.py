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
circle_small_contour = get_largest_contour(circle_small, factor=0)

def get_contours(image, factor=3):
    """
    Returns the contours of an image according to a mean-threshold.
    """
    _, image_thresh = cv2.threshold(
        image, image.mean() + image.std()*factor, 255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(image_thresh, 1, 2)
    return contours

def get_largest_contour(image, contours=None, factor=3):
    """
    Returns largest contour of the contour list.

    Args:
        image (np.ndarray): Image to extract the contours from.
    Returns:
        np.ndarray. First element of contours list which lists the boundaries 
            of the contour.

    Method is making an implicit assumption that there will only be one
    contour (beam) in the image. 
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

    Kwargs:
        image (np.ndarray): Image to calculate moments from.
        contour (np.ndarray): Beam contour boundaries.
    Returns:
        list. List of zero, first and second image moments for x and y.

    Attempts to find the moments using an inputted contours first, but if it
    isn't inputted it will compute the contours of the image then compute
    the moments.
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

    Args:
        M (list): Moments of an image.
    Returns:
        tuple. Centroid of the image.
    """    
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def get_bounding_box(image=None, contour=None):
    """
    Finds the up-right bounding box that contains the inputted contour.

    Kwargs:
        image (np.ndarray): Image to get a bounding box for.
        contour (np.ndarray): Beam contour boundaries.
    Returns:
        tuple. Contains x, y, width, height of bounding box.

    It should be noted that the x and y coordinates are for the bottom left
    corner of the bounding box. Use matplotlib.patches.Rectangle to plot.
    """
    try:
        return cv2.boundingRect(contour)
    except TypeError:
        contour, _ = get_largest_contour(image)
        return cv2.boundingRect(contour)

def get_circularity(contour, method=1):
    """
    Returns a score of how circular a contour is by comparing it to the template
    image, "circle_small.png." in the template_images directory.
    """
    return cv2.matchshapes(circle_small_contour, contour, method, 0)
