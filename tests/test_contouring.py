#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for functions in the preprocessing module.
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
from psbeam.images.templates import (circle, lenna)
from psbeam.contouring import (get_contours)
from psbeam.preprocessing import threshold_image
from psbeam.beamexceptions import NoContoursDetected

# get_contours

def test_get_contours_returns_correct_contours():
    circle_thr = threshold_image(circle, mode="mean")
    _, circle_cnt_cv2, _ = cv2.findContours(circle_thr, 1, 2)
    circle_cnt_psb = get_contours(circle, thresh_mode="mean", factor=1)
    for cnt_cv2, cnt_psb in zip(circle_cnt_cv2, circle_cnt_psb):
        assert(cnt_cv2.all() == cnt_psb.all())

def test_get_contours_raises_nocontoursdetected_when_no_contours():
    test_image = np.zeros((100,100), dtype=np.uint8)
    with pytest.raises(NoContoursDetected):
        get_contours(test_image)

# get_largest_contour

# def test_
        
