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
from psbeam.contouring import (get_contours, get_largest_contour, get_moments)
from psbeam.preprocessing import (to_gray, threshold_image)
from psbeam.beamexceptions import NoContoursDetected

# get_contours

def test_get_contours_returns_correct_contours():
    circle_thr = threshold_image(circle, mode="mean")
    _, circle_cnts_cv2, _ = cv2.findContours(circle_thr, 1, 2)
    circle_cnts_psb = get_contours(circle, thresh_mode="mean", factor=1)
    for cnts_cv2, cnts_psb in zip(circle_cnts_cv2, circle_cnts_psb):
        assert(cnts_cv2.all() == cnts_psb.all())

def test_get_contours_raises_nocontoursdetected_when_no_contours():
    test_image = np.zeros((100,100), dtype=np.uint8)
    with pytest.raises(NoContoursDetected):
        get_contours(test_image)

# get_largest_contour

def test_get_largest_contour_returns_largest_contour_of_image():
    lenna_thr = threshold_image(to_gray(lenna), mode="mean")
    _, lenna_cnts_cv2, _ = cv2.findContours(lenna_thr, 1, 2)
    lenna_cnts_area = np.array([cv2.contourArea(cnt) for cnt in lenna_cnts_cv2])
    lenna_largest_area_cv2 = lenna_cnts_area.max()
    lenna_largest_cnt_cv2 = lenna_cnts_cv2[np.argmax(lenna_cnts_area)]
    lenna_largest_cnt_psb, lenna_largest_area_psb = get_largest_contour(
        image=to_gray(lenna), thresh_mode="mean")
    assert(lenna_largest_area_cv2 == lenna_largest_area_psb)
    assert(lenna_largest_cnt_cv2.all() == lenna_largest_cnt_psb.all())

def test_get_largest_contour_returns_largest_contour_of_contours():
    lenna_thr = threshold_image(to_gray(lenna), mode="mean")
    _, lenna_cnts_cv2, _ = cv2.findContours(lenna_thr, 1, 2)
    lenna_cnts_area = np.array([cv2.contourArea(cnt) for cnt in lenna_cnts_cv2])
    lenna_largest_area_cv2 = lenna_cnts_area.max()
    lenna_largest_cnt_cv2 = lenna_cnts_cv2[np.argmax(lenna_cnts_area)]
    lenna_largest_cnt_psb, lenna_largest_area_psb = get_largest_contour(
        contours=lenna_cnts_cv2, thresh_mode="mean")
    assert(lenna_largest_area_cv2 == lenna_largest_area_psb)
    assert(lenna_largest_cnt_cv2.all() == lenna_largest_cnt_psb.all())

# get_moments

def test_get_moments_returns_correct_moments_of_image():
    circle_largest_cnt, _ = get_largest_contour(image=circle,
                                                thresh_mode="mean")
    moments_cv2 = cv2.moments(circle_largest_cnt)
    moments_psb = get_moments(image=circle)
    for m in moments_cv2.keys():
        assert(m in moments_psb.keys())
        assert(moments_cv2[m] == moments_psb[m])

def test_get_moments_returns_correct_moments_of_contour():
    circle_largest_cnt, _ = get_largest_contour(image=circle,
                                                thresh_mode="mean")
    moments_cv2 = cv2.moments(circle_largest_cnt)
    moments_psb = get_moments(contour=circle_largest_cnt)
    for m in moments_cv2.keys():
        assert(m in moments_psb.keys())
        assert(moments_cv2[m] == moments_psb[m])
        

