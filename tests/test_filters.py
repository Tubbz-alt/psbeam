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
from psbeam.images import testing
from psbeam.images.templates import circle
from psbeam.images.testing import (beam_image_01, beam_image_02, beam_image_03,
                                   beam_image_04)
from psbeam.beamexceptions import NoBeamDetected
from psbeam.filters import (contour_area_filter, full_filter)

from psbeam.preprocessing import (threshold_image)
from psbeam.contouring import (get_contours, get_largest_contour)

# contour_area_filter

def test_contour_area_filter_returns_false_for_empty_images():
    image_passes = contour_area_filter(np.zeros((10,10),dtype=np.uint8))
    assert(image_passes == False)

def test_contour_area_filter_returns_false_for_small_areas():
    array_test = np.zeros((10,10),dtype=np.uint8)
    array_test[:,0] = 1
    image_passes = contour_area_filter(array_test, min_area=20)
    assert(image_passes == False)
    
def test_contour_area_filter_correctly_filters_beam_images():
    beam_images = [beam_image_01, beam_image_02, beam_image_03, beam_image_04]
    for image in beam_images:
        image_passes = contour_area_filter(image)
        if image is beam_image_04:
            assert(image_passes == False)
        else:
            assert(image_passes == True)
