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

def detect(image, resize=1.0, kernel=(11,11), thresh_mode="otsu",
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

