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
from psbeam.preprocessing import (to_gray, threshold_image)
from psbeam.images.templates import (circle, lenna)
from psbeam.morph import get_opening, get_closing
from psbeam.beamexceptions import InputError

# get opening

def test_get_opening_correctly_morphs_image():
    lenna_bw = threshold_image(lenna)
    n_erode = 5
    n_dilate = 5
    kernel = (5,5)

    for i in range(n_erode):
        for j in range(n_dilate):
            lenna_eroded = cv2.erode(lenna_bw, kernel, iterations=i)
            lenna_opened = cv2.dilate(lenna_eroded, kernel, iterations=i)
            assert(lenna_opened.all() == get_opening(
                lenna_bw, n_erode=i, n_dilate=j, kernel=kernel).all())

def test_get_opening_raises_inputerror_on_non_binary_image():
    with pytest.raises(InputError):
        get_opening(np.arange(10))

# get_closing

def test_get_closing_correctly_morphs_image():
    lenna_bw = threshold_image(lenna)
    n_erode = 5
    n_dilate = 5
    kernel = (5,5)

    for i in range(n_erode):
        for j in range(n_dilate):
            lenna_dilated = cv2.dilate(lenna_bw, kernel, iterations=j)
            lenna_closed = cv2.erode(lenna_dilated, kernel, iterations=i)
            assert(lenna_closed.all() == get_closing(
                lenna_bw, n_erode=i, n_dilate=j, kernel=kernel).all())

def test_get_closing_raises_inputerror_on_non_binary_image():
    with pytest.raises(InputError):
        get_closing(np.arange(10))
        
