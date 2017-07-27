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
import numpy as np

##########
# Module #
##########
from psbeam.preprocessing import (to_uint8, uint_resize_gauss, threshold_image)

N = 2
ATOL = 1
dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint16, np.uint32,
          np.uint64, np.float16, np.float32, np.float64]
to_uint8_modes = ["clip", "norm", "scale"]

def test_to_uint8_converts_arrays_to_uint8_dtype():
    for dtype in dtypes:
        array_test = np.zeros((N,N), dtype=dtype)
        assert(array_test.dtype != np.uint8)
        for mode in to_uint8_modes:
            array_mode = to_uint8(array_test, mode)
            assert(array_mode.dtype == np.uint8)

def test_to_uint8_clip_mode_transforms_arrays_correctly():
    array_test = np.array([[-1000, 1000],[100, 200]])
    array_clip = to_uint8(array_test, mode="clip")
    array_expected = np.array([[0, 255],[100, 200]])
    assert(np.isclose(array_clip, array_expected, atol=ATOL).all())

def test_to_uint8_norm_mode_transforms_arrays_correctly():
    array_expected = np.array([[0, 255],[64, 191]])
    for dtype in dtypes:
        array_test = np.array([[0, 8],[2, 6]], dtype=dtype)
        array_norm = to_uint8(array_test, mode="norm")
        assert(np.isclose(array_norm, array_expected, atol=ATOL).all())
        
def test_to_uint8_scale_mode_transforms_arrays_correctly():
    # Values that will be tested are [[max_val, min_val, .25*range, .75*range]]
    array_expected = np.array([[0, 255],[64, 191]])
    for dtype in dtypes:
        try:
            # Grab array info for int types
            type_min = np.iinfo(dtype).min
            type_max = np.iinfo(dtype).max
        except ValueError:
            # Grab array info for float types
            type_min = np.finfo(dtype).min
            type_max = np.finfo(dtype).max
            
        array_range_half = type_max/2 - type_min/2
        # Clunky definition but gets around the problem of generating numbers
        # that are too large
        array_test = np.array([[type_min, type_max],
                               [type_min + np.round(array_range_half*.25) +
                                np.round(array_range_half*.25),
                                type_min + np.round(array_range_half*.75) +
                                np.round(array_range_half*.75)]],
                              dtype=dtype)
        # import ipdb; ipdb.set_trace()
        array_scale = to_uint8(array_test, mode="scale")
        assert(np.isclose(array_scale, array_expected, atol=ATOL).all())


