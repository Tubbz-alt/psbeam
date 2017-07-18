#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
############
# Standard #
############
import os
import logging

###############
# Third Party #
###############
import cv2
import numpy as np
import simplejson as sjson
from pyadplugin import ADPluginServer, ADPluginFunction

##########
# Module #
##########
from .statistics import contouring_pipeline

logger = logging.getLogger(__name__)

def contouring_plugin(ad_prefix, plugin_prefix="", plugin_suffix="", 
                      save_image=False, image_dir=None, save_json=False, 
                      json_path=None, min_cbtime=2, stream="IMAGE2",
                      enable_callbacks=True, resize=1.0, kernel=(11,11),
                      description="", threshold_factor=2):
    """
    Runs a pyadplugin that uses the contouring pipeline.
    """
    # Set the image saving path
    if save_image:
        save_frequency = 0.2
        if image_dir is None:
            image_dir = Path(os.path.dirname(os.path.abspath(__file__)) / 
                              "{0}_images_{1}".format(
                                  plugin_prefix, plugin_suffix))
        else:
            image_dir = Path(str(image_dir))
        # Check that the path exists, create it if not
        if not image_dir.exists():
            image_dir.mkdir(parents=True)

    # Set the json saving path
    if save_json:
        if json_path is None:
            json_path = Path(os.path.dirname(os.path.abspath(__file__)) / 
                             "{0}_data{1}.json".format(
                                 plugin_prefix, plugin_suffix))
        else:
            json_path  = Path(str(json_path))
        # Check the file and its parents exist, making them if they don't
        if not json_path.exists():
            json_path.parent.mkdir(parents=True)
            json_path.touch()

    # Description to be passed on as a PV
    if not description:
        description = "PyADPlugin '{0}{1}': Pipeline to output beam statitics."

    # Define the ADPluginFunction
    def pyad_contouring_plugin(array, height=None, width=None):
        return contouring_pipeline(
            array, height=height, width=width, resize=resize, kernel=kernel,
            prefix=plugin_prefix, suffix=plugin_suffix, save=save_frequency,
            description=description, json_path=json_path, save_image=save_image,
            image_dir=image_dir, thresh_factor=threshold_factor)

    # Define the default values for the pv dictionary
    output_dict = {
        "{0}:DESC{1}".format(plugin_prefix, plugin_suffix): description,
        "{0}:BEAM{1}".format(plugin_prefix, plugin_suffix) : False, 
        "{0}:CENT:X{1}".format(plugin_prefix, plugin_suffix) : -1, 
        "{0}:CENT:Y{1}".format(plugin_prefix, plugin_suffix) : -1, 
        "{0}:LENGTH{1}".format(plugin_prefix, plugin_suffix) : -1, 
        "{0}:WIDTH{1}".format(plugin_prefix, plugin_suffix) : -1, 
        "{0}:AREA{1}".format(plugin_prefix, plugin_suffix) : -1, 
        "{0}:MATCH{1}".format(plugin_prefix, plugin_suffix) : -1,
        "{0}:M{1}".format(plugin_prefix, plugin_suffix) : np.zeros((24))-1,
        }

    logger.info("Running '{0}{1}' server for '{2}'.".format(
        plugin_prefix, plugin_suffix, ad_prefix))

    try:
        # Set up the server
        pyad_server = ADPluginServer(
            prefix = prefix,
            ad_prefix = ad_prefix,
            stream = stream,
            min_cbtime = min_cbtime,
            enable_callbacks = enable_callbacks,
            )

        # Define the function
        pyad_function = ADPluginFunction(
            "{0}{1}".format(plugin_prefix, plugin_suffix), 
            output_dict,
            pyad_contouring_plugin,
            pyad_server,
            )
        
    # Log any exceptions we run into
    except Exception as e:
        logger.error("Exception raised by pyad server/function:\n{0}".format(e))
        raise

    
    

    
            
        
            
            
