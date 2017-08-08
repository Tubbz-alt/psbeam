"""
OpenCV Bluesky Plans
"""
############
# Standard #
############
import time
import logging

###############
# Third Party #
###############
import cv2
import numpy as np
from pswalker.plans import measure

##########
# Module #
##########
from ..filters import contour_area_filter
from ..utils import (to_image, signal_tuple)
from ..beamexceptions import (NoContoursDetected, InputError)
from ..preprocessing import uint_resize_gauss
from ..contouring import (get_largest_contour, get_moments, get_centroid,
                          get_contour_size, get_similarity)

logger = logging.getLogger(__name__)


# List of characteristics in the return dictionary

stat_list = ["sum_mn_raw",          # Mean of the sum of raw image
             "sum_std_raw",         # Std of sum of the raw image
             "sum_mn_prep",         # Mean of the sum of preprocessed image
             "sum_std_prep",        # Std of sum of the preprocessed image
             "mean_mn_raw",         # Mean of the mean of raw image
             "mean_std_raw",        # Std of the mean of raw_image
             "mean_mn_prep",        # Mean of the mean of preprocessed image
             "mean_std_prep",       # Std of the mean of preprocessed image
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

def process_det_data(data, detector, size_signal, kernel=(13,13),
                     uint_mode="scale", thresh_mode="otsu", thresh_factor=3,
                     resize=1.0, md=None, **kwargs):
    """
    Processes each image in the dict and returns another dict with the
    processed data.
    """
    stats_array = np.zeros((len(data), 10))
    for i, d in enumerate(data):
        # Array of processed image data for each shot in a dict for each det
        stats_array[i,:] = process_image(
            to_image(d[detector.name], size_signal=size_signal), kernel=kernel,
            resize=resize, uint_mode=uint_mode, thresh_mode=thresh_mode,
            **kwargs)

    # Remove any rows that have -1 as a value
    stats_array_dropped = np.delete(stats_array, np.unique(np.where(
        stats_array == -1)[0]), axis=0)

    # Turn the data into a mean and std for each entry                
    results_dict = dict()
    for i in range(stats_array_dropped.shape[1]):
        results_dict[stat_list[2*i]] = stats_array_dropped[:,i].mean()
        results_dict[stat_list[2*i+1]] = stats_array_dropped[:,i].std()

    # Meta-Data
    if md is None or md.lower() == "none":
        pass    
    # Basic meta-data
    elif md.lower() == "basic":
        results_dict["md"] = {
            "len_data" : len(data),
            "dropped" : len(data) - len(stats_array_dropped)}
        
    # All computed data - debugging purposes only!
    elif md.lower() == "all":
        logger.warning("Meta-Data is set to 'all'")
        results_dict["md"] = {
            "len_data" : len(data),
            "dropped" : len(data) - len(stats_array_dropped),
            "doc" : data,
            "stats_array" : stats_array,
            "dropped_array" : stats_array_dropped,
        }

    else:
        logger.warning("Invalid meta-data mode entry '{0}'. Valid modes are "
                       "'basic', 'all' or None. Skipping.".format(md))
    return results_dict

def characterize(detector, array_signal_str, size_signal_str, num=10,
                 filters=None, delay=None, drop_missing=True, kernel=(9,9),
                 resize=1.0, uint_mode="scale", min_area=100, md=None,
                 thresh_factor=3, filter_kernel=(9,9), thresh_mode="otsu",
                 **kwargs):
    """
    Returns a dictionary containing all the relevant statistics of the beam.
    """   
    # Get the image and size signals
    array_signal = getattr(detector, array_signal_str)
    size_signal = getattr(detector, size_signal_str)
    
    # Apply the default filter
    if filters is None:
        filters = dict()
    array_signal_str_full = detector.name + "_" + array_signal_str.replace(
        ".", "_")
    filters[array_signal_str_full] = lambda image : contour_area_filter(
        to_image(image, detector, size_signal))
    
    # Get images for all the shots
    data = yield from measure([array_signal], num=num, delay=delay,
                              filters=filters, drop_missing=drop_missing)
    
    # Process the data    
    results = process_det_data(data, array_signal, size_signal, kernel=kernel,
                               uint_mode=uint_mode, thresh_mode=thresh_mode,
                               thresh_factor=thresh_factor, resize=resize,
                               md=md, **kwargs)
                                    
    return results


