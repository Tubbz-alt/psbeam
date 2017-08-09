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
from psbeam.beamdetector import detect
from psbeam.images.testing.dg3 import images as images_dg3
from psbeam.images.testing.hx2 import images as images_hx2

beam_images = [beam_image_01, beam_image_02, beam_image_03, beam_image_04]

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
    for i, image in enumerate(beam_images):
        image_passes = contour_area_filter(image)
        if image is beam_image_04:
            assert(image_passes == False)
        else:
            assert(image_passes == True)

def test_contour_area_filter_works_on_dg3_images():
    for key, img in images_dg3.items():
        image_passes = contour_area_filter(img, kernel=(19,19), factor=4)
        assert(image_passes == bool(int(key[-3:])) and True)

def test_contour_area_filter_works_on_hx2_images():
    for key, img in images_hx2.items():
        image_passes = contour_area_filter(img, kernel=(19,19), factor=2)
        assert(image_passes == bool(int(key[-3:])) and True)             

# full_filter

def test_full_filter_returns_true():
    image_passes = full_filter(circle, (127,127), cent_atol=1000,
                               thresh_m00_min=-10e9, thresh_m00_max=10e9,
                               thresh_similarity=1000, kernel=(3,3))
    assert(image_passes == True)
    
def test_full_filter_returns_false_on_moment_too_high():
    image_passes = full_filter(circle, (127,127), cent_atol=100,
                               thresh_m00_min=-10e9, thresh_m00_max=-10e8,
                               thresh_similarity=1000, kernel=(3,3))
    assert(image_passes == False)

def test_full_filter_returns_false_on_moment_too_low():
    image_passes = full_filter(circle, (127,127), cent_atol=1000,
                               thresh_m00_min=10e8, thresh_m00_max=10e9,
                               thresh_similarity=1000, kernel=(3,3))
    assert(image_passes == False)

def test_full_filter_returns_false_on_centriods_too_different():
    image_passes = full_filter(circle, (0,0), cent_atol=1,
                               thresh_m00_min=-10e9, thresh_m00_max=10e9,
                               thresh_similarity=1000, kernel=(3,3))
    assert(image_passes == False)

def test_full_filter_returns_false_on_contour_too_dissimilar():
    semi_circle = np.copy(circle)
    semi_circle[:, :127] = 0
    image_passes = full_filter(semi_circle, (127,127), cent_atol=100,
                               thresh_m00_min=-10e9, thresh_m00_max=10e9,
                               thresh_similarity=0.001, kernel=(3,3))
    assert(image_passes == False)
    
def test_full_filter_returns_true_correctly():
    for i, image in enumerate(beam_images):
        cent, _ = detect(image)
        image_passes = full_filter(image, cent, n_opening=0, kernel=(3,3),
                                   thresh_m00_min=5, thresh_m00_max=100)
        if image is beam_image_04:
            assert(image_passes == False)
        else:
            assert(image_passes == True)

def test_full_filter_works_on_hx2_images():
    for key, img in images_hx2.items():
        try:
            cent, _ = detect(img, kernel=(9,9))
        except NoBeamDetected:
            cent = (0,0)
        image_passes = full_filter(img, cent, kernel=(9,9), n_opening=3,
                                   thresh_similarity=0.1, cent_atol=3)        
        assert(image_passes == bool(int(key[-3:])) and True)             
            
def test_full_filter_works_on_dg3_images():
    for key, img in images_dg3.items():
        try:
            cent, _ = detect(img, kernel=(9,9))
        except NoBeamDetected:
            cent = (0,0)
        image_passes = full_filter(img, cent, kernel=(9,9), n_opening=3,
                                   thresh_similarity=0.1, cent_atol=3)
        assert(image_passes == bool(int(key[-3:])) and True)
            

