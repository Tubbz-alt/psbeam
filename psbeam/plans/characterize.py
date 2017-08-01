"""
OpenCV Bluesky Plans
"""
############
# Standard #
############
import time
import logging
from multiprocessing import Pool
from functools import partial

###############
# Third Party #
###############
import cv2
import bluesky
import numpy as np
from pswalker.plans import measure
# from ophyd import Device, Signal
# from bluesky.utils import Msg
# from bluesky.plans import mv, trigger_and_read, run_decorator, stage_decorator

##########
# Module #
##########
from ..filters import contour_area_filter
from ..beamexceptions import (NoContoursDetected, InputError)
from ..preprocessing import (to_uint8, uint_resize_gauss)
from ..contouring import (get_largest_contour, get_moments, get_centroid,
                          get_contour_size, get_similarity)

logger = logging.getLogger(__name__)

# List of characteristics in the return dictionary

stat_list = ["raw_sum_mn",          # Mean of the sum of raw image
             "raw_sum_std",         # Std of sum of the raw image
             "prep_sum_mn",         # Mean of the sum of preprocessed image
             "prep_sum_std",        # Std of sum of the preprocessed image
             "raw_mean_mn",         # Mean of the mean of raw image
             "raw_mean_std",        # Std of the mean of raw_image
             "prep_mean_mn",        # Mean of the mean of preprocessed image
             "prep_mean_std",       # Std of the mean of preprocessed image
             "area_mn",             # Mean of area of the beam
             "area_std",            # Std of area of the beam
             "centroid_x_mn",       # Mean of centroid x
             "centroid_x_std",      # Std of centroid x
             "centroid_y_mn",       # Mean of centroid y
             "centroid_y_std",      # Std of centroid y
             "length_mn",           # Mean of the beam length
             "length_std",          # Std of the beam length
             "width_mn",            # Mean of the beam width
             "width_std",           # Std of the beam width
             "match_mn",            # Mean beam similarity score
             "match_std"]           # Std beam similarity score

def to_image(array, detectors=[], size_signal=None, shape=None):
    """
    Tries to convert the inputted array into an image format.
    """
    # Check that we can get an image shape
    if size_signal and detectors:
        sizes = [list(int(val) for val in getattr(det, size_signal).get())
             for det in detectors]
        if not all(size == sizes[0] for size in sizes):
            raise InputError("Multiple sizes found for detectors")
        shape = sizes[0]
    elif not shape:
        raise InputError("Must input either a signal and detector or expected "
                         "array shape.")
    
    # Check the shape value
    if all(val == 0 for val in shape):
        raise ValueError('Invalid image shape; ensure array_callbacks are on')
    if shape[-1] == 0:
        shape = shape[:-1]
    
    return to_uint8(array.reshape(shape), mode="clip")

def process_image(image, resize=1.0, kernel=(13,13), uint_mode="scale",
                  thresh_mode="otsu", thresh_factor=3):
    """
    Processes the input image and returns an array of numbers charcterizing the
    beam.

    Parameters
    ----------
    image : np.ndarray
        Image to process

    resize : float, optional
        Resize the image before performing any processing.

    kernel : tuple, optional
        Size of kernel to use when running the gaussian filter.

    factor : int, float
    	Factor to pass to the mean threshold.

    Returns
    -------
    np.ndarray
    	Array containing all the relevant fields of the image    
    """
    # Preprocess with a gaussian filter
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, kernel=kernel,
                                   mode=uint_mode)
    
    # The main pipeline
    try:
        contour, area = get_largest_contour(image_prep, thesh_mode=thresh_mode,
                                            factor=thresh_factor)
        M = get_moments(contour=contour)
        centroid_y, centroid_x = [pos//resize for pos in get_centroid(M)]
        l, w = [val//resize for val in get_contour_size(contour=contour)]
        match = get_similarity(contour)

    # No beam on Image, set values to make this clear
    except NoContoursDetected:
        area = -1
        centroid_y, centroid_x = [-1, -1]
        l = -1
        w = -1   
        match = -1

    # Basic info
    mean_raw = image.mean()
    mean_prep = image_prep.mean()
    sum_raw = image.sum()
    sum_prep = image_prep.sum()
    
    return np.array([sum_raw, sum_prep, mean_raw, mean_prep, area, centroid_x,
                     centroid_y, l, w, match])

def process_det_data(data, detectors, det_sizes, kernel=(13,13), uint_mode="scale",
                     thresh_mode="otsu", thresh_factor=3, resize=1.0):
    """
    Processes each image in the dict and returns another dict with the
    processed data.
    """
    result = dict()
    # image_data = {det.name : np.zeros((len(data), 10)) for det in detectors}
    for size, det in zip(det_sizes, detectors):
        stats_array = np.zeros((len(data), 10))
        for i, d in enumerate(data):
            # Array of processed image data for each shot in a dict for each det
            stats_array[i,:] = process_image(
                to_image(d[det.name], shape=size), kernel=kernel,
                resize=resize, uint_mode=uint_mode, thresh_mode=thresh_mode)

        # Remove any rows that have -1 as a value
        stats_array_dropped = np.delete(stats_array, np.unique(np.where(
            stats_array == -1)[0]), axis=0)

        # Turn the data into a mean and std for each entry                
        results_dict = dict()
        for i in range(stats_array_dropped.shape[1]):
            results_dict[stat_list[2*i]] = stats_array_dropped[:,i].mean()
            results_dict[stat_list[2*i+1]] = stats_array_dropped[:,i].std()

        # Key the array by det name
        result[det.name] = results_dict
        
    return result

def characterize(detectors, image_signal, size_signal, num=10, filters=None,
                 delay=None, drop_missing=True, kernel=(9,9), resize=1.0,
                 uint_mode="scale", min_area=100, thresh_factor=3, filter_kernel=(9,9),
                 thresh_mode="otsu", **kwargs):
    """
    Returns a dictionary containing all the relevant statistics of the beam.
    """
    # Apply the default filter
    if filters is None:
        filters = dict()
    for det in detectors:
        array_str = det.name + "_" + image_signal.replace(".", "_")
        filters[array_str] = lambda image : contour_area_filter(
            to_image(image, detectors, size_signal))
    # Get the image signals
    image_signals = [getattr(det, image_signal) for det in detectors]
    det_sizes = [list(int(val) for val in getattr(det, size_signal).get())
                   for det in detectors]
    
    # Get images for all the shots
    data = yield from measure(image_signals, num=num, delay=delay,
                              filters=filters, drop_missing=drop_missing)
    
    # Process the data    
    results = process_det_data(data, image_signals, det_sizes, kernel=kernel,
                               uint_mode=uint_mode, thresh_mode=thresh_mode,
                               thresh_factor=thresh_factor, resize=resize)
                                    
    return results


