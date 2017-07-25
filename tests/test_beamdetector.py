#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for functions in the beamdetector module
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
from psbeam.beamexceptions import NoBeamDetected
from psbeam.beamdetector import detect

# detect

def test_detect_raises_nobeamdetected_when_no_beam():
    with pytest.raises(NoBeamDetected):
        detect(np.zeros((50,50)))

def test_detect_finds_center_of_circle():
    centroid, bounding_box = detect(circle)
    assert(centroid == [s/2-1 for s in circle.shape])
