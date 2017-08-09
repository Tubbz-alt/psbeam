#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for functions in the filters script.
"""
############
# Standard #
############
import logging
from pprint import pprint
from time import time

###############
# Third Party #
###############
import cv2
import pytest
import numpy as np
from bluesky.utils import Msg
from bluesky.plans import run_wrapper
from pswalker.plans import measure

##########
# Module #
##########
from .utils import collector
from .conftest import beam_images
from psbeam.plans.characterize import (process_image, process_det_data,
                                       characterize)
from psbeam.beamdetector import detect
from psbeam.beamexceptions import (NoBeamDetected, NoContoursDetected)
from psbeam.filters import contour_area_filter
from psbeam.preprocessing import uint_resize_gauss
from psbeam.contouring import (get_largest_contour, get_moments, get_centroid,
                               get_contour_size, get_similarity)

def test_sim_det_device_interfaces_with_psbeam_properly(sim_det_01):
    det = sim_det_01
    im_filter = lambda image : contour_area_filter(image, uint_mode="clip")
    for i in range(10):
        try:
            image = det.image.image
            idx = det.image.array_counter.value % 4
            cent, bbox = detect(image, uint_mode="clip", filters=im_filter)
            assert(idx != 3 or i == 0)
        except NoBeamDetected:
            assert(idx == 0)
        
def test_sim_det_interfaces_with_bluesky_correctly(sim_det_01, sim_det_02, RE):
    global_data = None
    det_01 = sim_det_01
    det_02 = sim_det_02
    im_filter = lambda image : contour_area_filter(image, uint_mode="clip")

    # Fake event storage
    array_data = []
    array_count = []
    col_images = collector(det_01.image.array_data.name, array_data)
    col_count = collector(det_01.image.array_counter.name, array_count)

    # A test plan that just reads the data 10 times
    def test_plan(det):
        read_data = yield from measure([det], num=10)
        
    # Include the counter in the read
    det_01.image.read_attrs = ["array_data", "array_counter"]

    # Run the plan
    RE(run_wrapper(test_plan(det_01)), subs={'event':[col_count, col_images]})

    # Check the each image against the count
    im_filter = lambda image : contour_area_filter(image, uint_mode="clip")
    for count, array in zip(array_count, array_data):
        try:                
            array_size = [int(val) for val in det_01.image.array_size.get()]
            if array_size == [0, 0, 0]:
                raise RuntimeError('Invalid image')
            if array_size[-1] == 0:
                array_size = array_size[:-1]

            image = np.array(array).reshape(array_size)
            idx = count - 1
            cent, bbox = detect(image, uint_mode="clip", filters=im_filter)
            assert(idx % 4 != 3 or idx == 0)
        except NoBeamDetected:
            assert(idx % 4 == 3)
    
def test_process_image_returns_correct_arrays():
    resize = 1.0
    kernel = (3,3)
    uint_mode = "clip"
    thresh_mode = "otsu"
    thresh_factor = 3
    
    for image in beam_images:
        process_array = process_image(image, kernel=kernel, uint_mode=uint_mode,
                                      thresh_mode=thresh_mode, resize=resize,
                                      thresh_factor=thresh_factor)
        # Check the shape is as expected
        assert(process_array.shape == (10,))

        # Perform the same process
        try:
            image_prep = uint_resize_gauss(image, fx=resize, fy=resize,
                                           kernel=kernel)
            contour, area = get_largest_contour(
                image_prep, factor=thresh_factor, thesh_mode=thresh_mode)
            M = get_moments(contour=contour)
            centroid_y, centroid_x = [pos//resize for pos in get_centroid(M)]
            l, w = [val//resize for val in get_contour_size(contour=contour)]
            match = get_similarity(contour)
            
        except NoContoursDetected:
            area = -1
            centroid_y, centroid_x = [-1, -1]
            l = -1
            w = -1   
            match = -1
        mean_raw = image.mean()
        mean_prep = image_prep.mean()
        sum_raw = image.sum()
        sum_prep = image_prep.sum()
        
        # Put it all together
        test_array = np.array([mean_raw, mean_prep, sum_raw, sum_prep, area,
                               centroid_x, centroid_y, l, w, match])
        assert(process_array.all() == test_array.all())
        
def test_process_det_data_returns_correct_detector_dict(sim_det_01, RE):
    resize = 1.0
    kernel = (9,9)
    uint_mode = "clip"
    thresh_mode = "otsu"
    thresh_factor = 3        
    det_01 = sim_det_01

    # Fake event storage
    array_data = []
    array_count = []
    col_images = collector(det_01.image.array_data.name, array_data)
    col_count = collector(det_01.image.array_counter.name, array_count)    

    # A test plan that reads the data 10 times and process the data
    def test_plan(det):
        read_data = yield from measure([det], num=10)
        array_signal = det.image.array_data 
        size_signal = det.image.array_size
        
        proc_dict = process_det_data(read_data, array_signal, size_signal,
                                     kernel=kernel, resize=resize,
                                     uint_mode=uint_mode,
                                     thresh_mode=thresh_mode)
        assert(len(proc_dict) == 20)
        
    # Run the plan
    RE(run_wrapper(test_plan(det_01)),
       subs={'event':[col_count, col_images]})

def test_characterize_returns_correct_detector_dict(sim_det_01, RE):
    resize = 1.0
    kernel = (9,9)
    uint_mode = "scale"
    thresh_mode = "otsu"
    thresh_factor = 3
    min_area = 100
    det_01 = sim_det_01
    array_str = "image.array_data"
    size_str = "image.array_size"

    # Fake event storage
    array_data = []
    array_count = []
    col_images = collector(det_01.image.array_data.name, array_data)
    col_count = collector(det_01.image.array_counter.name, array_count)    

    # A test plan that reads the data 10 times and process the data
    def test_plan(det, sig, size):
        proc_dict = yield from characterize(
            det, array_str, size_str, num=10, kernel=kernel, resize=resize,
            uint_mode=uint_mode, min_area=min_area, thresh_factor=thresh_factor)
        assert(len(proc_dict) == 20)
        
    # Run the plan
    RE(run_wrapper(test_plan(det_01, "image.array_data", "image.array_size")),
       subs={'event':[col_count, col_images]})
    
