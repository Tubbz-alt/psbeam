#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions that run a beam statistics pipeline returning a dictionary with the
calculated values from the inputted image.
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
                        kernel=(13,13), prefix="", suffix="", save=0.1, desc="", 
                        json_path=None, save_image=None, save_image_path=None,
                        thresh_factor=2):
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
    """
    # Resize back into an image
    image = np.reshape(array, (height, width))
    # Preprocess with a gaussian filter
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, 
                                   kernel=kernel)

    # The main pipeline
    try:
        contour, area = get_largest_contour(image_prep, factor=factor)
        M = get_moments(contour=contour)
        centroid = [pos//resize for pos in get_centroid(M)]
        l, w = [val//resize for val in get_contour_size(contour, factor=factor)]
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
        "{0}:DESC":desc,
        "{0}:BEAM".format(prefix) : beam_present, 
        "{0}:CENT:X".format(prefix) : centroid[0], 
        "{0}:CENT:Y".format(prefix) : centroid[1], 
        "{0}:LENGTH".format(prefix) : l, 
        "{0}:WIDTH".format(prefix) : w, 
        "{0}:AREA".format(prefix) : area, 
        "{0}:MATCH".format(prefix) : match,
        "{0}:M".format(prefix) : np.array([M[m] for m in sorted(M.keys())]),
           }
    
    # Saving as a json file 
    if json_path is not None:
        if not json_path.exists():
            json_path.touch()
        with file_path.open(mode='a') as json:
            sjson.dump(output, json, indent=4)
    
    # Save the image if save_image and a path is passed
    if save_image is not None and save_image_path is not None:
        # Save at random according to save_image
        if np.random.uniform < save_image:
            # imwrite returns False if it fails to save
            if not cv2.imwrite(save_image_path, image):
                # Make sure the parent folders exist, before trying again
                save_image_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_image_path, image)

    return output
