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
from .statistics import contouring_pipeline

logger = logging.getLogger(__name__)

def stats_01(array, height=None, width=None, resize=1.0, kernel=(13,13)):
    desc = "Gauss filter for prep. Use largest contour. Use sum for beam " \
      "presence."
    file_str = Path("{0}.json".format(ad_prefix.replace(":", "_")))
    return contouring_pipeline(array, height=height, width=width,
                               prefix="PYSTATS:", suffix=":01",
                               description=desc, save_image=True) 

def stats_02(array, height=None, width=None, resize=1.0, kernel=(13,13)):
    desc = "Gauss filter then 2 erosions and 2 dilations for prep. Use " \
      "largest contour. No check for beam presence."
    return contouring_pipeline(array, height=height, width=width,
                               prefix="PYSTATS:", suffix=":02",
                               desccription=desc, save_image=True)

# Set up the server
ad_prefix = 'HX2:SB1:CVV:01:'
server = ADPluginServer(prefix='PYSTATS:',
                        ad_prefix=ad_prefix,
                        stream='IMAGE2',
                        min_cbtime=2,
                        enable_callbacks=True)

# Set the file that takes the json file
file_str = "{0}".format(ad_prefix.replace(":", "_"))
json_path = Path("/reg/g/pcds/pyps/apps/skywalker/json/{0}.json".format(file_str))
image_path = Path("/reg/g/pcds/pyps/apps/skywalker/images/{0}.png".format(file_str))

# Set up the plugins
stats01 = ADPluginFunction("01", {"PYSTATS:AREA:01": 0, 
                                  "PYSTATS:BEAM:01": False, 
                                  "PYSTATS:CENT:X:01": 0, 
                                  "PYSTATS:CENT:Y:01": 0, 
                                  "PYSTATS:LENGTH:01":0, 
                                  "PYSTATS:WIDTH:01":0, 
                                  "PYSTATS:DESC:01":"",
                                  "PYSTATS:MATCH:01":0,
                                  "PYSTATS:M:01":np.zeros((24))-1}, 
                         stats_01, server)
stats02 = ADPluginFunction("02", {"PYSTATS:AREA:02": 0, 
                                  "PYSTATS:BEAM:02": False, 
                                  "PYSTATS:CENT:X:02": 0, 
                                  "PYSTATS:CENT:Y:02": 0, 
                                  "PYSTATS:LENGTH:02": 0, 
                                  "PYSTATS:WIDTH:02": 0, 
                                  "PYSTATS:DESC:02": "",
                                  "PYSTATS:MATCH:02": 0,
                                  "PYSTATS:M:02":np.zeros((24))-1}, 
                           stats_02, server)

# import ipdb; ipdb.set_trace()

