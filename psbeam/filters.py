#!/usr/bin/env python
"""
Signal filters used in Skywalker.
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
from .morph import get_opening
from .preprocessing import uint_resize_gauss
from .beamexceptions import (NoContoursDetected, NoBeamDetected,
                             MomentOutOfRange)
from .contouring import (get_largest_contour, get_moments, get_centroid,
                         get_similarity)

logger = logging.getLogger(__name__)

def contour_area_filter(image, kernel=(9,9), resize=1.0, uint_mode="scale",
                        min_area=100, factor=3, **kwargs):
    """
    Checks that a contour can be returned for two thresholds of the image.

    Parameters
    ----------
    image : np.ndarray
    	Image to check for contours

    kernel : tuple, optional
    	Kernel to use when gaussian blurring.

    resize : float, optional
    	How much to resize the image by before doing any calculations.

    uint_mode : str, optional
    	Conversion mode to use when converting to uint8

    min_area : float, optional
    	Minimum area of the otsu thresholded beam.

    factor : int, float
    	Factor to pass to the mean threshold.

    Returns
    -------
    passes : bool
    	True if the image passes the check, False if it does not
    """
    image_prep = uint_resize_gauss(image, mode=uint_mode, kernel=kernel,
                                   fx=resize, fy=resize)
    # Try to get contours of the image
    try:
        contour_mean, area_mean = get_largest_contour(
            image_prep, thresh_mode="mean", factor=factor, **kwargs)        
        contour_otsu, area_otsu = get_largest_contour(
            image_prep, thresh_mode="otsu", **kwargs)
        # Do the check for area
        if area_otsu < min_area:
            return False
        return True
    except NoContoursDetected:
        return False        

def full_filter(image, centroids_ad, resize=1.0, kernel=(13,13), n_opening=2,
                cent_rtol=0.1, threshold_m00_min=50, threshold_m00_max=10e6,
                threshold_similarity=0.067):
    """
    Runs the full pipeline which includes:
        - Checks if there is beam by obtaining an image contour
        - Checks the sum of all pixels is above and below a threshold
        - Checking if the computed centroid is close to the adplugin centroid
        - Checks that the beam is above the threshold of similarity

    Parameters
    ----------
    image : np.ndarray
        Image to process

    cbentroids_ad : tuple
        Centroids obtained from the areadetector stats plugin.

    resize : float, optional
        Resize the image before performing any processing.

    kernel : tuple, optional
        Size of kernel to use when running the gaussian filter.

    n_opening : int, optional
        Number of times to perform an erosion, followed by the same number of
        dilations.

    cent_rtol : float, optional
        Relative tolerance to use when comparing AD's and OpenCV's centroids.

    threshold_m00_min : float, optional
        Lower threshold for the sum of pixels in the image.

    threshold_m00_max : float, optional
        Upper threshold for the sum of pixels in the image.

    threshold_cicularity : float, optional
        Upper threshold for beam similarity score (0.0 is perfectly circular).

    Returns
    -------
    bool
        Bool indicating whether the image passed the tests.
    """
    try:
        # # Pipeline
        # Preprocessing
        image_prep = uint_resize_gauss(image, fx=resize, fy=resize, 
                                       kernel=kernel)
        # Morphological Opening
        image_morph = get_opening(image_prep, n_erode=n_opening, 
                                  n_dilate=n_opening)
        # Grab the image contours
        contours = get_contours(image_morph)
        # Grab the largest contour
        contour, area = get_largest_contour(contours=contours)
        # Image moments
        M = get_moments(contour=contour)
        # Find a centroid
        centroids_cv2 = [pos//resize for pos in get_centroid(M)]
        # Get a score for how similar the beam contour is to a circle's contour
        similarity = get_similarity(contour)
        
        # # Filters
        # Sum of pixel intensities must be between m00_min and m00_max
        if M['m00'] < threshold_m00_min or M['m00'] > threshold_m00_max:
            logger.debug("Filter - Image sum ouside specified range. Sum: "
                         "{0}".format(M['m00']))
            return False
        
        # The centroids of both ad and cv must be close
        for cent_ad, cent_cv in zip(centroids_ad, centroids_cv):
            if not np.isclose(cent_ad, cent_cv, rtol=cent_rtol):
                logger.debug("Filter - AD and OpenCV centroids not close. "
                             "AD Centroid: {0} OpenCV Centroid: {1}".format(
                                 centroids_ad, centroids_cv))
                return False
            
        # Check that the similarity of the beam is below the inputted threshold
        if similarity > threshold_similarity:
            logger.debug("Filter - Beam cicularity too low. Cicularity: "
                         "{0}".format(similarity))
            return False
        
        # Everything passes
        logger.debug("Filter - Passed all filters with sum: {0}, OpenCV "
                     "centroids: {1}, AD Centroids: {2}, and similarity: {3}"
                     "".format(M['m00'], cent_cv, cent_ad, similarity))
        return True
    
    except NoContoursDetected:
        # Failed to get image contours
        logger.debug("Filter - No contours found on image.")
        return False

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
        if not min_m0 <= M['m00'] <= max_m0:
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
    
