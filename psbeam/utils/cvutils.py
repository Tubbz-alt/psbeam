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

################################################################################
#                                Image Operations                              #
################################################################################

def to_uint8(image, mode="clip"):
    """*Correctly* converts an image to uint8 type.
    
    Args:
        image (np.ndarray): Image to be converted to uint8.
    Returns:
        np.ndarray. Converted Image.
        
    Running 'image.astype(np.uint8)' on its own applies a mod(256) to handle
    values over 256. The correct way is to either clip (implemented here) or
    normalize.
    """
    if not isinstance(image, np.ndarray):
        image_array = np.array(image)
    else:
        image_array = image
    if mode == "clip":
        np.clip(image_array, 0, 255, out=image_array)
    elif mode == "norm":
        image_array *= 255/image_array.max()
    else:
        raise ValueError
    return image_array.astype(np.uint8)

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

################################################################################
#                                   Image I/O                                  #
################################################################################

def get_images_from_dir(target_dir, n_images=None, shuffle=False, out_type=list,
                        recursive=False, read_mode=cv2.IMREAD_GRAYSCALE,
                        glob="*"):
    """
    Crawls through the contents of inputted directoryb and saves files with 
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

    
        
    

