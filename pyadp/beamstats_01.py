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
import simplejson as sjson
from pathlib import Path

##########
# Module #
##########
from psbeam.beamexceptions import NoBeamPresent
import psbeam.beamdetector as psb
import psbeam.preprocessing as prep

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Globals
circ = cv2.imread("/reg/neh/home5/apra/work/python/psbeam/psbeam/images" \
                  "/circle_small.png", 0)
circ_cnt = psb.get_contours(circ, factor=0)[0]


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
        # psb.beam_is_present(M=M)
        centroid = [pos//resize for pos in psb.get_centroid(M)]
        _, _, l, w = [val//resize for val in psb.get_bounding_box(
            image_prep, contour)]
        match = cv2.matchShapes(circ_cnt, contour, 1, 0.0)
        beam_present = True
        l = l/255 * image.shape[0]
        w = w/255 * image.shape[1]
    except NoBeamPresent:
        beam_present = False
        area = 0
        centroid = [0,0]
        l = 0
        w = 0    
        match=0
        contour = None
        M = None
    res = {"01:AREA": area, "01:BEAM": beam_present, "01:CENT:X": centroid[0], 
           "01:CENT:Y": centroid[1], "01:LENGTH":l, "01:WIDTH":w, "01:DESC":desc,
           "01:MATCH":match }
    extra = {"M":M, "CNT":contour}
    with file_path.open(mode='a') as json:
        sjson.dump({**res, **extra}, json, indent=4)
    return res

def stats_02(array, height=None, width=None, resize=1.0, kernel=(13,13)):
    desc = "Gauss filter then 2 erosions and 2 dilations for prep. Use " \
      "largest contour. No check for beam presence."
    # Resize back into an image
    image = np.reshape(array, (height, width))
    # Preprocess with a gaussian filter
    image_prep = prep.uint_resize_gauss(image, kernel=kernel)
    
    try:
        image_morph = psb.get_opening(image_prep, erode=3, dilate=3)
        _, contours, _ = cv2.findContours(image_morph, 1, 2)
        contour, area = psb.get_largest_contour(image_prep, contours=contours, 
                                                get_area=True)
        M = psb.get_moments(contour=contour)
        # Check if beam is in the image using the sum of the pixel values
        # psb.beam_is_present(M=M)
        centroid = [pos//resize for pos in psb.get_centroid(M)]
        _, _, l, w = [val//resize for val in psb.get_bounding_box(
            image_prep, contour)]
        match = cv2.matchShapes(circ_cnt, contour, 1, 0.0)
        beam_present = True
        l = l/255 * image.shape[0]
        w = w/255 * image.shape[1]
    except NoBeamPresent:
        beam_present = False
        area = 0
        centroid = [0,0]
        l = 0
        w = 0
        match=0
        contour = None
        M = None
    res = {"02:AREA": area, "02:BEAM": beam_present, "02:CENT:X": centroid[0], 
           "02:CENT:Y": centroid[1], "02:LENGTH":l, "02:WIDTH":w, "02:DESC":desc,
           "02:MATCH":match}
    extra = {"M":M, "CNT":contour}
    with file_path.open(mode='a') as json:
        sjson.dump({**res, **extra}, json, indent=4)
    return res

# Set up the server
ad_prefix = 'HX2:SB1:CVV:01:'
server = ADPluginServer(prefix='PYSTATS:',
                        ad_prefix=ad_prefix,
                        stream='IMAGE2',
                        min_cbtime=2,
                        enable_callbacks=True)

# Set the file that takes the json file
file_str = "{0}extra.json".format(ad_prefix.replace(":", "_"))
file_path = Path("/reg/g/pcds/pyps/apps/skywalker/json/{0}".format(file_str))

# Check it exists
if not file_path.exists():
    file_path.touch()

# Set up the plugins
stats01 = ADPluginFunction("01", {"01:AREA": 0, "01:BEAM": False, "01:CENT:X": 0, 
                                "01:CENT:Y": 0, "01:LENGTH":0, "01:WIDTH":0, "01:DESC":"",
                                "01:MATCH":0}, 
                         stats_01, server)
stats02 = ADPluginFunction("02", {"02:AREA": 0, "02:BEAM": False, "02:CENT:X": 0, 
                                  "02:CENT:Y": 0, "02:LENGTH":0, "02:WIDTH":0, "02:DESC":"",
                                  "02:MATCH":0}, 
                           stats_02, server)

# import ipdb; ipdb.set_trace()

