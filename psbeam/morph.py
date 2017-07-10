#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions and classes related to performing morphological operations using
OpenCV.

For more information on Morphological Transformations, see OpenCV's 
documentation:

http://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
"""
############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2

##########
# Module #
##########

def get_opening(image, n_erode=1, n_dilate=1, kernel=np.ones((5,5),np.uint8)):
    """
    Performs the specified number of erosions, followed by the specified number
    of dilations to get the opening of the image.
    """
    image_eroded = cv2.erode(image, kernel, iterations=n_erode)
    image_opened = cv2.dilate(image_erode, kernel, iterations=n_dilate)
    return image_opened

def get_closing(image, n_erode=1, n_dilate=1, kernel=np.ones((5,5),np.uint8)):
    """
    Performs the specified number of dilations, followed by the specified number
    of erosions to get the closing of the image.
    """
    image_dilated = cv2.dilate(image, kernel, iterations=n_dilate)
    image_closed = cv2.erode(image_dilated, kernel, iterations=n_erode)
    return image_closed

