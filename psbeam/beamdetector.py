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
from .utils import isiterable
from .preprocessing import uint_resize_gauss
from .beamexceptions import (NoContoursDetected, NoBeamDetected, InputError)
from .contouring import (get_largest_contour, get_moments, get_bounding_box, 
                         get_centroid)

logger = logging.getLogger(__name__)

def detect(image, resize=1.0, kernel=(9,9), thresh_mode="otsu",
           uint_mode="scale", thresh_factor=1, filters=None, **kwargs):
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

    thresh_mode : str, optional
        Thresholding mode to use. For extended documentation see
        preprocessing.threshold_image. Valid modes are:
            ['mean', 'top', 'bottom', 'adaptive', 'otsu']

    uint_mode : str, optional
        Conversion mode to use when converting to uint8. Valid modes are:
            ['clip', 'norm', 'scale']

    factor : int or float, optional
        Factor to pass to the binary threshold.

    filters : callable or iterable, optional
        Beam filters that are run before attempting to find the centroids and
        bounding box. If the function returns False, NoBeamDetected is raised
    
    Returns
    -------
    tuple
        Tuple of centroid and bounding box

    Raises
    ------
    NoBeamDetected
        If there was no beam found on the image or if a filter returns False

    InputError
        If an invalid filter type is passed
    """
    # Skip if filters is none
    if filters is None:
        pass
    # Run if it is a function
    elif callable(filters):
        image_passes = filters(image)
        if not image_passes:
            raise NoBeamDetected        
    # Loop through each filter if it's a list
    elif isiterable(filters):
        for f in filters:
            if not f(image):
                raise NoBeamDetected
    # Invalid type inputted
    else:
        raise InputError("Invalid filter input type. Must be callable or "
                         "iterable. Got {0} instead.".format(type(filters)))

    # Begin detection pipeline
    image_prep = uint_resize_gauss(image, mode=uint_mode, fx=resize, fy=resize,
                                   kernel=kernel)
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

