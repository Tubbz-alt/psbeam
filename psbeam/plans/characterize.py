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
    Processes the input image and returns a vector of numbers charcterizing the
    beam.

    Parameters
    ----------
    image : np.ndarray
        Image to process

    resize : float, optional
        Resize the image before performing any processing.

    kernel : tuple, optional
        Size of kernel to use when running the gaussian filter.

    uint_mode : str, optional
    	Conversion mode to use when converting to uint8. For extended
    	documentation, see preprocessing.to_uint8. Valid modes are:
    		['clip', 'norm', 'scale']

    thresh_mode : str, optional
    	Thresholding mode to use. For extended documentation see
    	preprocessing.threshold_image. Valid modes are:
    		['mean', 'top', 'bottom', 'adaptive', 'otsu']

    thresh_factor : int, float
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

def process_det_data(data, array_signal, size_signal, resize=1.0, kernel=(13,13),
                     uint_mode="scale", thresh_mode="otsu", thresh_factor=3,
                     md=None, **kwargs):
    """
    Processes each image in the inputted event docs and returns another dict
    with the beam statistics calculated for all the events.

    Parameters
    ----------
    data : list
    	The list of event doc dicts that contain the array_data to be processed.

    array_signal : ophyd.signal.Signal
    	Signal that emitted the array data we will be processing.

    size_signal : ophyd.signal.Signal
    	Detector signal that returns the expected size of the array.

    resize : float, optional
        Resize the image before performing any processing.

    kernel : tuple, optional
        Size of kernel to use when running the gaussian filter.

    uint_mode : str, optional
    	Conversion mode to use when converting to uint8. For extended
    	documentation, see preprocessing.to_uint8. Valid modes are:
    		['clip', 'norm', 'scale']

    thresh_mode : str, optional
    	Thresholding mode to use. For extended documentation see
    	preprocessing.threshold_image. Valid modes are:
    		['mean', 'top', 'bottom', 'adaptive', 'otsu']

    thresh_factor : int or float, optional
    	Factor to pass to the mean threshold.

    md : str, optional
    	How much meta data to include in the output dict. Valid options are:
    		[None, 'basic', 'all']
    	Note: The 'all' option is for debugging purposes and should not be used
    	in production.

    Returns
    -------
    results_dict : dict
    	Dictionary containing the statistics obtained from the array data, and
    	optionally some meta-data.
    """
    stats_array = np.zeros((len(data), 10))
    for i, event in enumerate(data):
        # Array of processed image data for each shot in a dict for each det
        stats_array[i,:] = process_image(
            to_image(event[array_signal.name], size_signal=size_signal),
            kernel=kernel, resize=resize, uint_mode=uint_mode,
            thresh_mode=thresh_mode, **kwargs)

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
                 delay=None, filters=None, drop_missing=True, 
                 filter_kernel=(9,9), resize=1.0, uint_mode="scale",
                 min_area=100, filter_factor=(9,9), min_area_factor=3,
                 kernel=(9,9), thresh_factor=3, thresh_mode="otsu", md=None,
                 **kwargs):
    """
    Characterizes the beam profile by computing various metrics and statistics
    of the beam using the inputted detector. The function performs 'num' reads
    on the array_data field of the detector, optionally filtering shots until
    'num' shots have been collected, and then runs the processing pipeline on
    each of the resulting arrays.

    The processing pipeline computes the contours of the image, from which the
    area, length, width, centroid and circularity of the contour is computed.
    Additionally, the sum and mean intensity values are computed of both the
    preprocessed image and the raw image.

    Once the pipeline has been finished processing for all 'num' images, the
    mean and standard deviation of each statistic is computed, giving a total
    20 entries to the stats dictionary.

    Computed Statistics
    -------------------
    sum_mn_raw
    	Mean of the sum of the raw image intensities.
    
    sum_std_raw
    	Standard deviation of sum of the raw image pixel intensities.
    
    sum_mn_prep
    	Mean of the sum of the preprocessed image pixel intensities. 
    
    sum_std_prep
    	Standard deviation of the sum of the preprocessed image pixel
    	intensities.
    
    mean_mn_raw
    	Mean of the mean of the raw image pixel intensities.
    
    mean_std_raw
    	Standard deviation of the mean of the raw image pixel intensities.
    
    mean_mn_prep
    	Mean of the mean of the preprocessed image pixel intensities.
    
    mean_std_prep
    	Standard deviation of the mean of the preprocessed image pixel
    	intensities.
    
    area_mn
    	Mean of the area of the contour of the beam.
    
    area_std
    	Standard deviation of area of the contour of the beam.
    
    centroid_x_mn
    	Mean of the contour centroid x.
    
    centroid_x_std
    	Standard deviation of the contour centroid x.
    
    centroid_y_mn
    	Mean of the contour centroid y.
    
    centroid_y_std
    	Standard deviation of the contour centroid y.
    
    length_mn
    	Mean of the contour length.
    
    length_std
    	Standard deviation of the contour length.
    
    width_mn
    	Mean of the contour width.
    
    width_std
    	Standard deviation of the contour width.
    
    match_mn
    	Mean score of contour similarity to a binary image of a circle.
    
    match_std
    	Standard deviation of score of contour similarity to a binary image of
    	a circle.

    Parameters
    ----------
    detector : detector obj
    	Detector object that contains the components for the array data and
    	image shape data.

    array_signal_str : str
    	The string name of the array_data signal.

    size_signal_str : str
    	The string name of the signal that will provide the shape of the image.
    
    num : int
        Number of measurements that need to pass the filters.

    delay : float
        Minimum time between consecutive reads of the detectors.

    filters : dict, optional
        Key, callable pairs of event keys and single input functions that
        evaluate to True or False. 

    drop_missing : bool, optional
        Choice to include events where event keys are missing.

    filter_kernel : tuple, optional
    	Kernel to use when gaussian blurring in the contour filter.

    resize : float, optional
    	How much to resize the image by before doing any calculations.

    uint_mode : str, optional
    	Conversion mode to use when converting to uint8.

    min_area : float, optional
    	Minimum area of the otsu thresholded beam.

    filter_factor : float
    	Factor to pass to the filter mean threshold.

    min_area_factor : float
    	The amount to scale down the area for comparison with the mean threshold
    	contour area.
    
    kernel : tuple, optional
        Size of kernel to use when running the gaussian filter.    

    thresh_mode : str, optional
    	Thresholding mode to use. For extended documentation see
    	preprocessing.threshold_image. Valid modes are:
    		['mean', 'top', 'bottom', 'adaptive', 'otsu']

    thresh_factor : int or float, optional
    	Factor to pass to the mean threshold.

    md : str, optional
    	How much meta data to include in the output dict. Valid options are:
    		[None, 'basic', 'all']
    	Note: The 'all' option is for debugging purposes and should not be used
    	in production.

    Returns
    -------
    results_dict : dict
    	Dictionary containing the statistics obtained from the array data, and
    	optionally some meta-data.    
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
        to_image(image, detector, size_signal), kernel=filter_kernel,
        min_area=min_area, min_area_factor=min_area_factor,
        factor=filter_factor, uint_mode=uint_mode)
    
    # Get images for all the shots
    data = yield from measure([array_signal], num=num, delay=delay,
                              filters=filters, drop_missing=drop_missing)
    
    # Process the data    
    results = process_det_data(data, array_signal, size_signal, kernel=kernel,
                               uint_mode=uint_mode, thresh_mode=thresh_mode,
                               thresh_factor=thresh_factor, resize=resize,
                               md=md, **kwargs)
                                    
    return results


