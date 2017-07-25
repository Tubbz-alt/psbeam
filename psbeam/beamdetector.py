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
from .beamexceptions import (NoContoursDetected, NoBeamDetected)
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
    NoBeamDetected
    """    
    try:
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise NoBeamDetected
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
            raise NoBeamDetected

def detect(image, resize=1.0, kernel=(11,11)):
    """
    Checks for beam presence and returns the centroid and bounding box 
    of the beam. Returns None if no beam is present.

    Args:
        image (np.ndarray): Image to find the beam on.
    Kwargs:
        resize (float): Resize factor (1.0 keeps image same size).
        kernel (tuple): Tuple of length 2 for gaussian kernel size.        
    Returns:
        tuple. Tuple of centroid and bounding box. None, None if no beam is
            present.
    """
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, kernel=kernel)
    try:
        contour, _ = get_largest_contour(image_prep)
        M = get_moments(contour=contour)
        centroid     = [pos//resize for pos in get_centroid(M)]
        bounding_box = [val//resize for val in get_bounding_box(contour)]
        return centroid, bounding_box
    except NoContoursDetected:
        raise NoBeamDetected

