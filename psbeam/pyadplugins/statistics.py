#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions that run a beam statistics pipeline, returning a dictionary with the
calculated values from the inputted image. These are put together with the
expectation that they will be used by the PyADPlugin package.
"""
############
# Standard #
############
import logging
from datetime import datetime

###############
# Third Party #
###############
import cv2
import numpy as np
import simplejson as sjson

##########
# Module #
##########
from ..morph import get_opening
from ..preprocessing import uint_resize_gauss
from ..beamexceptions import NoContoursPresent
from ..contouring import (get_largest_contour, get_moments, get_centroid,
                          get_circularity, get_contour_size)

logger = logging.getLogger(__name__)

def contouring_pipeline(array, height=None, width=None, resize=1.0, 
                        kernel=(11,11), prefix="", suffix="", save=0.2, 
                        description="", 
                        json_path=None, save_image=None, image_dir=None,
                        threshold_factor=2):
    """
    Runs a pipeline that returns:
        - DESC - Description of the plugin
        - BEAM - Beam presence
        - CENT X - Centroid in X
        - CENT Y - Centroid in Y
        - LENGTH - Length of the beam
        - WIDTH - Width of the beam
        - AREA - Area of the beam
        - MATCH - Circularity of the beam
        - M - Moments of the beam

    Returns
    -------
    output : dict
        Dictionary with next round of PV values.
    """
    # Reshape into an image
    image = np.reshape(array, (height, width))
    # Preprocess with a gaussian filter
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, 
                                   kernel=kernel)
    
    # The main pipeline
    try:
        contour, area = get_largest_contour(image_prep, factor=threshold_factor)
        M = get_moments(contour=contour)
        centroid = [pos//resize for pos in get_centroid(M)]
        l, w = [val//resize for val in get_contour_size(
            contour, factor=threshold_factor)]
        match = get_circularity(contours)
        beam_present = True

    # No beam on Image, set values to make this clear
    except NoContoursPresent:
        beam_present = False
        area = -1
        centroid = [-1,-1]
        l = -1
        w = -1   
        match=-1
        M = np.zeros((24)) - 1

    # Set the values in the dictionary
    output = {
        "{0}:DESC{1}".format(prefix, suffix): description,
        "{0}:BEAM{1}".format(prefix, suffix) : beam_present, 
        "{0}:CENT:X{1}".format(prefix, suffix) : centroid[0], 
        "{0}:CENT:Y{1}".format(prefix, suffix) : centroid[1], 
        "{0}:LENGTH{1}".format(prefix, suffix) : l, 
        "{0}:WIDTH{1}".format(prefix, suffix) : w, 
        "{0}:AREA{1}".format(prefix, suffix) : area, 
        "{0}:MATCH{1}".format(prefix, suffix) : match,
        "{0}:M{1}".format(prefix, suffix) : np.array([M[m] for m in sorted(
            M.keys())]),
           }
    
    # Saving as a json file 
    if json_path is not None:
        if not json_path.exists():
            json_path.touch()
        with file_path.open(mode='a') as json:
            sjson.dump(output, json, indent=4)
    
    # Save the image if save_image and a path is passed
    if save_image is not None and image_dir is not None:
        # Save at random according to save_image
        if np.random.uniform < save_image:
            # Create the save name for the image
            save_image_path = image_dir / "{0}_image_{1}.png".format(
                prefix, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
            # imwrite returns False if it fails to save
            if not cv2.imwrite(str(save_image_path), image):
                # Make sure the parent folders exist, before trying again
                logger.warn("Image path provided does not exist. Making " \
                            "parent(s) directory(ies).")
                save_image_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_image_path, image)

    return output
