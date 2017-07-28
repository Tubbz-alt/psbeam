#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for functions in the filters script.
"""
############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2
import pytest
import numpy as np

##########
# Module #
##########
from psbeam.beamdetector import detect
from psbeam.beamexceptions import NoBeamDetected
from psbeam.filters import contour_area_filter 

def test_sim_det_device_interfaces_properly(sim_det):
    det = sim_det
    im_filter = lambda image : contour_area_filter(image, uint_mode="clip")
    for i in range(10):
        try:
            image = det.image.image
            idx = det._n_image
            cent, bbox = detect(image, uint_mode="clip", filters=im_filter)
            assert(det._has_beam)
        except NoBeamDetected:
            assert(not det._has_beam)
        
    
