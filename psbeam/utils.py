"""
Image-Centric Helper Functions
"""
############
# Standard #
############
import os
import random
import logging
from pathlib import Path

###############
# Third Party #
###############
import cv2
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def isiterable(obj):
    """
    Function that determines if an object is an iterable, not including 
    str.

    Parameters
    ----------
    obj : object
        Object to test if it is an iterable.

    Returns
    -------
    bool : bool
        True if the obj is an iterable, False if not.
    """
    if isinstance(obj, str):
        return False
    else:
        return isinstance(obj, Iterable)

def rolling_average (values, window):
    weights = np.repeat(1.0, window)/window
    return np.convolve(values, weights, 'valid')

def plot_image(image,  msg = ""):
    """
    Plots an image with an optional message.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    if msg:
        plt.text(0.95, 0.05, msg, ha='right', va='center', color='w',
                transform=ax.transAxes)
    plt.grid()
    plt.show()

def get_images_from_dir(target_dir, n_images=None, shuffle=False, out_type=list,
                        recursive=False, read_mode=cv2.IMREAD_UNCHANGED,
                        glob="*"):
    """
    Crawls through the contents of inputted directory and saves files with 
    image extensions as images.
    """
    image_ext = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
    target_dir_path = Path(target_dir)
    if recursive and glob == "*":
        glob = "**"
    # Grab path of all image files in dir
    image_paths = [p for p in sorted(target_dir_path.glob(glob)) if
                   p.is_file() and p.suffix[1:].lower() in image_ext]
    
    # Shuffle the list of paths
    if shuffle:
        random.shuffle(image_paths)
        if out_type is dict:
            logger.warning("Shuffle set to True for requested output type dict")
    # Only keep n_images of those files
    if n_images:
        image_paths = image_paths[:n_images]

    # Return as the desired type
    if out_type is dict:
        return {p.stem : cv2.imread(str(p), read_mode) for p in image_paths}
    else:
        return out_type([cv2.imread(str(p), read_mode) for p in image_paths])
        
    

