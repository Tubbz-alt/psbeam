#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting functions to be used for and throughout psbeam.
"""
############
# Standard #
############
import logging

###############
# Third Party #
###############
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from multiprocessing import Process

##########
# Module #
##########

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
