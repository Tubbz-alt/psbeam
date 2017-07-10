#!/usr/bin/env python
# -*- coding: utf-8 -*-
############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from multiprocessing import Process

##########
# Module #
##########
from .beamexceptions import NoBeamPresent
from .utils.cvutils import to_uint8
from .preprocessing import uint_resize_gauss

def get_opening(image, erode=1, dilate=1, kernel=np.ones((5,5),np.uint8)):
    img_erode = cv2.erode(image, kernel, iterations=erode)
    img_dilate = cv2.erode(img_erode, kernel, iterations=dilate)
    return img_dilate


def get_contours(image, factor=3):
    """
    Returns the contours of an image according to a mean-threshold.
    """
    _, image_thresh = cv2.threshold(
        image, image.mean() + image.std()*factor, 255, cv2.THRESH_TOZERO)
    _, contours, _ = cv2.findContours(image_thresh, 1, 2)
    return contours

def get_largest_contour(image, contours=None, factor=3, get_area=False):
    """
    Returns largest contour of the contour list.

    Args:
        image (np.ndarray): Image to extract the contours from.
    Returns:
        np.ndarray. First element of contours list which lists the boundaries 
            of the contour.

    Method is making an implicit assumption that there will only be one
    contour (beam) in the image. 
    """
    if not contours:
        contours = get_contours(image, factor=factor)
    if not contours:
        raise NoBeamPresent
    area = [cv2.contourArea(cnt) for cnt in contours]
    if get_area:
        return contours[np.argmax(np.array(area))], np.array(area).max()

def get_moments(image=None, contour=None):
    """
    Returns the moments of an image.

    Kwargs:
        image (np.ndarray): Image to calculate moments from.
        contour (np.ndarray): Beam contour boundaries.
    Returns:
        list. List of zero, first and second image moments for x and y.

    Attempts to find the moments using an inputted contours first, but if it
    isn't inputted it will compute the contours of the image then compute
    the moments.
    """
    try:
        return cv2.moments(contour)
    except TypeError:
        contour = get_largest_contour(image)
        return cv2.moments(contour)

def get_centroid(M):
    """
    Returns the centroid using the inputted image moments.

    Centroid is computed as being the first moment in x and y divided by the
    zeroth moment.

    Args:
        M (list): Moments of an image.
    Returns:
        tuple. Centroid of the image.
    """    
    return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

def get_bounding_box(image=None, contour=None):
    """
    Finds the up-right bounding box that contains the inputted contour.

    Kwargs:
        image (np.ndarray): Image to get a bounding box for.
        contour (np.ndarray): Beam contour boundaries.
    Returns:
        tuple. Contains x, y, width, height of bounding box.

    It should be noted that the x and y coordinates are for the bottom left
    corner of the bounding box. Use matplotlib.patches.Rectangle to plot.
    """
    try:
        return cv2.boundingRect(contour)
    except TypeError:
        contour = get_largest_contour(image)
        return cv2.boundingRect(contour)


def beam_is_present(M=None, image=None, contour=None, max_m0=10e5, min_m0=10):
    """
    Checks if there is a beam in the image by checking the value of the 
    zeroth moment.

    Kwargs:
        M (list): Moments of an image. 
        image (np.ndarray): Image to check beam presence for.
        contour (np.ndarray): Beam contour boundaries.

    Returns:
        bool. True if beam is present, False if not.
    """    
    try:
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise BeamNotPresent
    except (TypeError, IndexError):
        if contour:
            M = get_moments(contour=contour)
        else:
            contour = get_largest_contour(image)
            M = get_moments(contour=contour)
        if not (M['m00'] < max_m0 and M['m00'] > min_m0):
            raise BeamNotPresent

def detect(image, resize=1.0, kernel=(11,11)):
    """
    Checks for beam presence and returns the centroid and bounding box 
    of the beam. Returns None if no beam is present.

    Args:
        image (np.ndarray): Image to find the beam on.
    Kwargs:
        resize (float): Resize factor (1.0 keeps image same size).
        kernel (tuple): Tuple of length 2 for gaussian kernel size.        
    Returns:
        tuple. Tuple of centroid and bounding box. None, None if no beam is
            present.
    """
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, kernel=kernel)
    contour = get_largest_contour(image_prep)
    M = get_moments(contour=contour)
    centroid, bounding_box = None, None
    if beam_is_present(M):
        centroid     = [pos//resize for pos in get_centroid(M)]
        bounding_box = [val//resize for val in get_bounding_box(image_prep, 
                                                                contour)]
    return centroid, bounding_box

def find(image):
    """
    Returns the centroid and bounding box of the beam.

    Args:
        image (np.ndarray): Image to find the beam on.
    Kwargs:
        resize (float): Resize factor (1.0 keeps image same size).
        kernel (tuple): Tuple of length 2 for gaussian kernel size.        
    Returns:
        tuple. Tuple of centroid and bounding box. None, None if no beam is
            present.

    This method assumes that beam is known to be present.
    """
    image_prep = uint_resize_gauss(image, fx=resize, fy=resize, kernel=kernel)
    contour = get_largest_contour(image_prep)
    M = get_moments(contour=contour)
    centroid     = [pos//resize for pos in get_centroid(M)]
    bounding_box = [val//resize for val in get_bounding_box(image_prep, 
                                                            contour)]
    return centroid, bounding_box


def _plot(image, centroids, bounding_boxes, msg):
    """
    Internal method. Plots the inputted image optionally with the 
    centroid, bounding box and text.

    Args:
        image (np.ndarray): Image to plot.
        centroid (tuple): X,y coordinates of the centroid.
        bounding_box (tuple): X,y,w,h of the up-right bounding box.
        msg (str): Text to display on bottom right of image.

    Handles multiple centroids or bounding boxes in one image by inputting a 
    list of tuples. Allows continuation of script execution using the Process
    object.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    for centroid, bounding_box in zip(centroids, bounding_boxes):
        if isinstance(centroid, tuple) and len(centroid) == 2:
            circ = plt.Circle(centroid, radius=5, color='g')
            ax.add_patch(circ)
            if not msg:
                msg = "Centroid: {0}".format(centroid)
        if isinstance(bounding_box, tuple) and len(tuple) == 4:
            x,y,w,h = bounding_box
            box = Rectangle((x,y),w,h,linewidth=2,edgecolor='r',
                            facecolor='none')
            ax.add_patch(box)
    if msg:
         plt.text(0.95, 0.05, msg, ha='right', va='center', color='w',
                  transform=ax.transAxes)
    plt.grid()
    plt.show()

def plot(image, centroid=[], bounding_box=[], msg="", wait=False):
    """
    Plots the inputted image optionally with the centroid, bounding box
    and text. Can halt execution or continue.

    Args:
        image (np.ndarray): Image to plot.
    Kwargs:
        centroid (tuple): X,y coordinates of the centroid.
        bounding_box (tuple): X,y,w,h of the up-right bounding box.
        msg (str): Text to display on bottom right of image.
        wait (bool): Halt script execution until image is closed.

    Handles multiple centroids or bounding boxes in one image by inputting a 
    list of tuples.
    """
    if isinstance(centroid, tuple):
        centroid = [centroid]
    if isinstance(bounding_box, tuple):
        bounding_box = [bounding_box]
    if wait:
        _plot(
            image, centroids=centroid, bounding_boxes=bounding_box, msg=msg)
    else:
        plot = Process(target=_plot, args=(image, centroid, bounding_box, msg))
        plot.start()


# class BeamException(Exception):
#     """
#     Base exception class for psbeam.
#     """
#     pass


# class NoBeamPresent(BeamException):
#     """
#     Exception raised if an operation requiring the beam is requested but no beam
#     is actually present.
#     """
#     def __init__(self, *args, **kwargs):
#         self.msg = kwargs.pop("msg", "Cannot perform operation; No beam found.")
#         super().__init__(*args, **kwargs)
#     def __str__(self):
#         return repr(self.msg)
