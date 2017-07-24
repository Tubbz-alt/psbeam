#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for functions in the preprocessing module.
"""
############
# Standard #
############
import pytest
import logging

###############
# Third Party #
###############
import cv2
import numpy as np

##########
# Module #
##########
from psbeam.beamexceptions import InputError
from psbeam.images.templates import lenna
from psbeam.preprocessing import (to_gray, to_uint8, uint_resize_gauss,
                                  threshold_image)

N = 2
ATOL = 1
dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint16, np.uint32,
          np.uint64, np.float16, np.float32, np.float64]
to_uint8_modes = ["clip", "norm", "scale"]

# to_gray

def test_to_gray_raises_inputerror_on_vectors():
    vector = np.arange(10)
    with pytest.raises(InputError):
        to_gray(vector)

def test_to_gray_raises_inputerror_on_gray_image():
    image_gray = np.random.rand(10,10).astype(np.uint8)
    with pytest.raises(InputError):
        to_gray(image_gray)

def test_to_gray_raises_inputerror_on_invalid_color_space():
    with pytest.raises(InputError):
        to_gray(lenna, color_space="test")

def test_to_gray_correctly_converts_from_rgb():
    lenna_gray_rgb_cv2 = cv2.cvtColor(lenna, cv2.COLOR_RGB2GRAY)
    assert(lenna_gray_rgb_cv2.all() == to_gray(lenna, color_space="RGB").all())

def test_to_gray_correctly_converts_from_bgr():
    lenna_gray_bgr_cv2 = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)
    assert(lenna_gray_bgr_cv2.all() == to_gray(lenna, color_space="BGR").all())

# to_uint8

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

# uint_resize_gausss
        
def test_uint_resize_gauss_converts_to_uint8():
    for dtype in dtypes:
        array_test = np.zeros((N,N), dtype=dtype)
        assert(array_test.dtype != np.uint8)
        array_prep = uint_resize_gauss(array_test)
        assert(array_prep.dtype == np.uint8)

def test_uint_resize_gauss_resizes_correctly():
    n = 10
    array_test = np.zeros((n, n))
    sizes = np.arange(0.1, 1, 0.1)
    for l, w in zip(sizes, sizes[::-1]):
        array_prep = uint_resize_gauss(array_test, fx=w, fy=l)
        assert(array_prep.shape == (int(l*n), int(w*n)))

def test_uint_resize_gauss_blurs_correctly():
    array_test = np.random.rand(N, N)
    kernel = (3,3)
    sigma = 0
    array_blur = cv2.GaussianBlur(array_test, kernel, sigma)
    array_prep = uint_resize_gauss(array_test)
    assert(array_blur.all() == array_prep.all())

# threshold_image

def test_threshold_image_mean_mode():
    factors = (3, -3)
    lenna_gray = to_gray(lenna)
    for factor in factors:
        _, lenna_thr_cv2 = cv2.threshold(lenna_gray, lenna_gray.mean() + \
                                         factor * lenna.std(), 255,
                                         cv2.THRESH_BINARY)
        lenna_thr_psb = threshold_image(lenna_gray, binary=True, mode="mean",
                                        factor=factor)
        assert(lenna_thr_cv2.all() == lenna_thr_cv2.all())

def test_threshold_image_top_mode():
    factors = (2, 3, 5)
    lenna_gray = to_gray(lenna)
    for factor in factors:
        _, lenna_thr_cv2 = cv2.threshold(lenna_gray, lenna_gray.max() - \
                                         factor * lenna.std(), 255,
                                         cv2.THRESH_BINARY)
        lenna_thr_psb = threshold_image(lenna_gray, binary=True, mode="top",
                                        factor=factor)
        assert(lenna_thr_cv2.all() == lenna_thr_cv2.all())

def test_threshold_image_bottom_mode():
    factors = (2, 3, 5)
    lenna_gray = to_gray(lenna)
    for factor in factors:
        _, lenna_thr_cv2 = cv2.threshold(lenna_gray, lenna_gray.min() + \
                                         factor * lenna.std(), 255,
                                         cv2.THRESH_BINARY)
        lenna_thr_psb = threshold_image(lenna_gray, binary=True, mode="bottom",
                                        factor=factor)
        assert(lenna_thr_cv2.all() == lenna_thr_cv2.all())

def test_threshold_image_adaptive_mode():
    lenna_gray = to_gray(lenna)
    block_size = 11
    c = 2
    lenna_thr_cv2 = cv2.adaptiveThreshold(lenna_gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, c)
    lenna_thr_psb = threshold_image(lenna_gray, binary=True, mode="adaptive",
                                    C=c, blockSize=block_size)
    assert(lenna_thr_cv2.all() == lenna_thr_cv2.all())

def test_threshold_image_adaptive_mode():
    lenna_gray = to_gray(lenna)
    _, lenna_thr_cv2 = cv2.threshold(lenna_gray, 0, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lenna_thr_psb = threshold_image(lenna_gray, binary=True, mode="otsu")
    assert(lenna_thr_cv2.all() == lenna_thr_cv2.all())
    
        
