#!/usr/bin/env python
# -*- coding: utf-8 -*-
############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2
import numpy as np
from pyadplugin import ADPluginServer, ADPluginFunction

##########
# Module #
##########
from psbeam.beamexceptions import NoBeamDetected
import psbeam.beamdetector as psb
import psbeam.preprocessing as prep

logger = logging.getLogger('pyadplugin.pyadplugin')
logger.setLevel(logging.DEBUG)

def stats_01(array, height=None, width=None, resize=1.0, kernel=(13,13)):
    desc = "Gauss filter for prep. Use largest contour. Use sum for beam " \
      "presence."
    # Resize back into an image
    image = np.reshape(array, (height, width))
    # Preprocess with a gaussian filter
    image_prep = prep.uint_resize_gauss(image, kernel=kernel)
    try:
        contours = psb.get_contours(image_prep)
        contour, area = psb.get_largest_contour(image_prep, contours=contours, 
                                                get_area=True)
        M = psb.get_moments(contour=contour)
        # Check if beam is in the image using the sum of the pixel values
        psb.beam_is_present(M=M)
        centroid = [pos//resize for pos in psb.get_centroid(M)]
        _, _, l, w = [val//resize for val in psb.get_bounding_box(
            image_prep, contour)]
        beam_present = True
    except NoBeamDetected:
        beam_present = False
        area = 0
        centroid = [0,0]
        l = 0
        w = 0
    res = {"AREA": area, "BEAM": beam_present, "CENT:X": centroid[0], 
           "CENT:Y": centroid[1], "LENGTH":l, "WIDTH":w, "DESC":desc}
    return res

# Set up the server
ad_prefix = 'HFX:DG3:CVV:01:'
server = ADPluginServer(prefix='PYSTATS:',
                        ad_prefix=ad_prefix,
                        stream='IMAGE2',
                        min_cbtime=10,
                        enable_callbacks=True)


# Set up the plugins
stats = ADPluginFunction("01", {"AREA": 0, "BEAM": False, "CENT:X": 0, 
                                "CENT:Y": 0, "LENGTH":0, "WIDTH":0, "DESC":""}, 
                         stats_01, server)

# import ipdb; ipdb.set_trace()

