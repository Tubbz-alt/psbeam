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

dir_path = "/reg/g/pcds/pyps/apps/skywalker/template_images/"

def save(array, height=None, width=None):
    image = np.reshape(array, (height, width))
    i = 0
    img_path = Path("{0}image_{1}.png".format(dir_path, i))
    pass
    # while img_path.exists():
    #     i += 1
    # cv2
    # return 0
