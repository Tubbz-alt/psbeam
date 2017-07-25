#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions that relate to beam detection such as checking for beam presence and
finding beam centroids.
"""
############
# Standard #
############
import logging

###############
# Third Party #
###############

##########
# Module #
##########
from .preprocessing import uint_resize_gauss
from .beamexceptions import (NoContoursDetected, NoBeamDetected,
                             MomentOutOfRange)
from .contouring import (get_largest_contour, get_moments, get_bounding_box, 
                         get_centroid)

def moments_within_range(M=None, image=None, contour=None, max_m0=10e5,
                         min_m0=10):
    """
    Checks that the image moments are within the specified range.

    Parameters
    ----------
    M : dict, optional 
    	Moments of the image.
    
    image : np.ndarray, optional
    	Image to check the moments for.
    
    contour : np.ndarray, optional
    	Beam contours

    max_m0 : float, optional
    	Maximum value that the zeroth moment can have

    min_m0 : float, optional
    	Minimum value that the zeroth moment can have

    Returns
    -------
    within_range : bool
    	Whether the moments were within the specified range.

    Raises
    ------
    MomentOutOfRange
    	If the zeroth moment is out of the specified range.
    """    
    try:
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise MomentOutOfRange
    except (TypeError, IndexError):
        if contour:
            M = get_moments(contour=contour)
        else:
            try:
                contour = get_largest_contour(image)
                M = get_moments(contour=contour)
            except NoContoursDetected:
                raise NoBeamDetected
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise MomentOutOfRange

def detect(image, resize=1.0, kernel=(11,11), thresh_mode="mean",
           thresh_factor=1):
    """
    Checks for beam presence and returns the centroid and bounding box 
    of the beam. Raises a NoBeamDetected exception if no beam is found.

    Parameters
    ----------
    image : np.ndarray
    	Image to detect the beam on
    
    resize : float, optional
    	Resize factor (1.0 keeps image same size).
    
    kernel : tuple, optional
    	Tuple of length 2 for gaussian kernel size.
    
    Returns
    -------
    tuple
    	Tuple of centroid and bounding box

    Raises
    ------
    NoBeamDetected
    	If there was no beam found on the image.
    """
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, kernel=kernel)
    try:
        contour, _ = get_largest_contour(image_prep, thresh_mode=thresh_mode,
                                         factor=thresh_factor)
        M = get_moments(contour=contour)
        centroid = [pos//resize for pos in get_centroid(M)]
        bounding_box = [val//resize for val in get_bounding_box(
            contour=contour)]
        return centroid, bounding_box
    except NoContoursDetected:
        raise NoBeamDetected

