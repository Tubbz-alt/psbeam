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
from ..beamexceptions import NoContoursDetected
from ..preprocessing import uint_resize_gauss
from ..contouring import (get_largest_contour, get_moments, get_centroid,
                          get_contour_size, get_similarity)

logger = logging.getLogger(__name__)



# def process_all_data(data, detectors, n_pool=None):
#     """
#     Process the data and return the results

#     Returns
#     -------
#     results : dict
#     	Dictionary of detector name to beam statistics
#     """
#     results_final = {}
#     pool = Pool(n_pool)
    
#     stat_list = ["sum_mn",         # Mean of the sum of pixels
#                  "sum_std",        # Std of sum of the pixels
#                  "mean_mn",        # Mean of the mean of pixels
#                  "mean_std",       # St of the mean of pixels
#                  "centroid_x_mn",  # Mean of centroid x
#                  "centroid_x_std", # Std of centroid x
#                  "centroid_y_mn",  # Mean of centroid y
#                  "centroid_y_std", # Std of centroid y
#                  "length_mn",      # Mean of the beam length
#                  "length_std",     # Std of the beam length
#                  "width_mn",       # Mean of the beam width
#                  "width_std",      # Std of the beam width
#                  "match_mn",       # Mean beam similarity score
#                  "match_std"]      # Std beam similarity score
    
#     for det in detectors:
#         results_det = {}
#         # Get all the data involved with this detector
#         all_stats = zip(*(pool.map(partial(process_det_data, detector=det),
#                                   data[det.name])))
#         for stat_key, stat in zip(stat_list, all_stats):
#             stat_array = np.array(stat)

# def to_image(array, detector):
#     """
#     Reshapes the inputted array according the image shapes specified by the
#     detector.
#     """
#     return np.reshape(np.array(array), (detector.cam.size.size_y.value,
#                                         detector.cam.size.size_x.value))

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

    mean_raw = image.mean()
    mean_prep = image_prep.mean()
    sum_raw = image.sum()
    sum_prep = image_prep.sum()
    
    return np.array([mean_raw, mean_prep, sum_raw, sum_prep, area, centroid_x,
                     centroid_y, l, w, match])

def process_det_data(data, detectors, kernel=(13,13), uint_mode="scale",
                     thresh_mode="otsu", thresh_factor=3, resize=1.0):
    """
    Processes each image in the dict and returns another dict with the
    processed data.
    """
    result = dict()
    image_data = {det.name : np.zeros((len(data), 10)) for det in detectors}
    for det in detectors:
        for i, d in enumerate(data):
            # Array of processed image data for each shot in a dict for each det
            image_data[det.name][i,:] = process_image(
                d[det.name].reshape((256,256)),
                kernel=kernel, resize=resize, uint_mode=uint_mode,
                thresh_mode=thresh_mode)

        # Turn the data into a mean and std for each entry
        result_array = np.zeros((image_data[det.name].shape[1] * 2))
        for i in range(image_data[det.name].shape[1]):
            result_array[2*i] = image_data[det.name][:,i].mean()
            result_array[2*i+1] = image_data[det.name][:,i].std()

        # Key the array by det name
        result[det.name] = result_array
        
    return result

def characterize(detectors, image_signal, num=10, filters=None, delay=None,
                 drop_missing=True, kernel=(9,9), resize=1.0, uint_mode="scale",
                 min_area=100, factor=3, n_pool=None, **kwargs):
    """
    Returns a dictionary containing all the relevant statistics of the beam.
    """
    # Apply the default filter
    if filters is None:
        filters = {image_signal : lambda image : contour_area_filter(
            to_image(image), kernel=kernel, resize=resize, uint_mode=uint_mode,
            min_area=min_area, factor=factor, **kwargs)}
    
    # Get images for all the shots
    data = yield from measure(getattr(detectors, image_signal), num=num,
                              delay=delay, filters=filters,
                              drop_missing=drop_missing)
    
    # Process the data    
    results = process_data(data, detectors, image_signal, n_pool=Pool)
                                    
    # return
    return results


# class Processor(object):
#     def __init__(self,resize=1.0, kernel=(13,13)):
#         self.resize=resize
#         self.kernel=kernel

#     def __call__(self, image):
#         image_prep = prep.uint_resize_gauss(image, kernel=self.kernel)
        
#         try:
#             contours = psb.get_contours(image_prep)

#             contour, area = psb.get_largest_contour(
#                 image_prep, contours=contours, get_area=True)
#             M = psb.get_moments(contour=contour)
#             # Check if beam is in the image using the sum of the pixel values
#             centroid = [pos//self.resize for pos in psb.get_centroid(M)]
#             _, _, l, w = [val//self.resize for val in psb.get_bounding_box(
#                 image_prep, contour)]

#             # import ipdb; ipdb.set_trace()

#             match = cv2.matchShapes(circ_cnt, contour, 1, 0.0)
#             return image.sum(), image.mean(), centroid[0], centroid[1], l, w, match
#         except NoBeamDetected:
#             return None, None, None, None, None, None, None

# def measure_beam_quality(detector, plugin, read_rate=None, num_reads=100):
#     """
#     Measures various stats about the beam.
#     """
#     if read_rate is None:
#         read_rate = min(plugin.array_rate.value/10, 10)
    
#     #Gather shots
#     # logger.debug("Gathering shots..")
#     # #Trigger detector and wait for completion
#     # yield Msg('trigger', detector, group='B')
#     # #Wait for completion
#     # yield Msg('wait', None, 'B')
#     # #Read outputs
#     # yield Msg('create', None, name='primary')
#     det_images = []
#     for i in range(num_reads):
#         time.sleep(1/read_rate)
#         det_images.append(plugin.image)

#     proc = Processor()
#     sums = np.zeros((len(det_images)))
#     means = np.zeros((len(det_images)))
#     centroids_x = np.zeros((len(det_images)))
#     centroids_y = np.zeros((len(det_images)))
#     lengths = np.zeros((len(det_images)))
#     widths = np.zeros((len(det_images)))
#     matches = np.zeros((len(det_images)))

#     for i, image in enumerate(det_images):
#         sums[i], means[i], centroids_x[i], centroids_y[i], lengths[i], widths[i], matches[i] = proc(
#             image)

#     results = {"sum_mn" : sums.mean(),
#                "sum_std" : sums.std(),
#                "mean_mn" : means.mean(),
#                "mean_std" : means.std(),
#                "centroid_x_mn" : centroids_x.mean(),
#                "centroid_x_std" : centroids_x.std(),
#                "centroid_y_mn" : centroids_y.mean(),
#                "centroid_y_std" : centroids_y.std(),
#                "length_mn" : lengths.mean(),
#                "length_std" : lengths.std(),
#                "width_mn" : widths.mean(),
#                "width_std" : widths.std(),
#                "match_mn" : matches.mean(),
#                "match_std" : matches.std()}
#     # yield Msg('save')
#     return results

#     # proc = Processor()
#     # pool = multiprocessing.Pool()
#     # # sums, means, centroids_x, centroids_y, lengths, widths, matches = pool.map(
#     # #     proc, det_images)
#     # results = pool.map(
#     #     proc, det_images)

        
#     # # sums = np.array(sums)
#     # # means = np.array(means)
#     # # centroids_x = np.array(centroids_x)
#     # # centroids_y = np.array(centroids_y)
#     # # lengths = np.array(lengths)
#     # # widths = np.array(widths)
#     # # matches = np.array(matches)

#     # # results = {"sum_mn" : sums.mean(),
#     # #            "sum_std" : sums.std(),
#     # #            "mean_mn" : means.mean(),
#     # #            "mean_std" : means.std(),
#     # #            "centroid_x_mn" : centroids_x.mean(),
#     # #            "centroid_x_std" : centroids_x.std(),
#     # #            "centroid_y_mn" : centroids_y.mean(),
#     # #            "centroid_y_std" : centroids_y.std(),
#     # #            "length_mn" : lengths.mean(),
#     # #            "length_std" : lengths.std(),
#     # #            "width_mn" : widths.mean(),
#     # #            "width_std" : widths.std(),
#     # #            "match_mn" : matches.mean(),
#     # #            "match_std" : matches.std()}
#     # return results


# if __name__ == "__main__":
#     # import pcdsdevices.epics.pim as pim
#     # p = pim.PIM("HFX:DG3:CVV:01")
#     # p = pim.PIM("HX2:SB1:CVV:01")

#     import pcdsdevices.sim.pim as pim
#     p = pim.PIM("TEST")
#     img = cv2.imread("../tests/test_yag_images_01/image97.jpg", 0)
#     p.detector.image1.image = img

#     res = measure_beam_quality(p.detector, p.detector.image1, read_rate=100, num_reads=10)

