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
from .preprocessing import uint_resize_gauss
from .contouring import (get_largest_contour, get_moments, get_bounding_box)

                         
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
