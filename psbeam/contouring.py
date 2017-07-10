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
from .beamexceptions import NoBeamPresent

def get_contours(image, factor=3):
    """
    Returns the contours of an image according to a mean-threshold.
    """
    _, image_thresh = cv2.threshold(
        image, image.mean() + image.std()*factor, 255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(image_thresh, 1, 2)
    return contours

def get_largest_contour(image, contours=None, factor=3, get_area=False):
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
    if not contours:
        contours = get_contours(image, factor=factor)
    if not contours:
        raise NoBeamPresent
    area = [cv2.contourArea(cnt) for cnt in contours]
    if get_area:
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
        contour = get_largest_contour(image)
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
        contour = get_largest_contour(image)
        return cv2.boundingRect(contour)


def beam_is_present(M=None, image=None, contour=None, max_m0=10e5, min_m0=10):
    """
    Checks if there is a beam in the image by checking the value of the 
    zeroth moment.

    Kwargs:
        M (list): Moments of an image. 
        image (np.ndarray): Image to check beam presence for.
        contour (np.ndarray): Beam contour boundaries.

    Returns:
        bool. True if beam is present, False if not.
    """    
    try:
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise BeamNotPresent
    except (TypeError, IndexError):
        if contour:
            M = get_moments(contour=contour)
        else:
            contour = get_largest_contour(image)
            M = get_moments(contour=contour)
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise BeamNotPresent
