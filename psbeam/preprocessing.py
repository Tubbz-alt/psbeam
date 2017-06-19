# Preprocessing functions used in psbeam
############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def to_uint8(image, mode="clip"):
    """
    *Correctly* converts an image to uint8 type.
    
    Args:
        image (np.ndarray): Image to be converted to uint8.
    Returns:
        np.ndarray. Converted Image.
        
    Running 'image.astype(np.uint8)' on its own applies a mod(256) to handle
    values over 256. The correct way is to either clip (implemented here) or
    normalize.
    """
    # Make sure the image is a numpy array
    if not isinstance(image, np.ndarray):
        image_array = np.array(image)
    else:
        image_array = image
    # Clip or normalize the image
    if mode.lower() == "clip":
        np.clip(image_array, 0, 255, out=image_array)
    elif mode.lower() == "norm":
        image_array *= 255/image_array.max()
    else:
        raise ValueError("Valid modes are 'clip' and 'norm'")
    return image_array.astype(np.uint8)

def uint_resize_gauss(image, mode='clip', fx=1.0, fy=1.0, kernel=(11,11), 
                      sigma=0):
    """
    Preprocess the image by converting to uint8, resizing and running a 
    gaussian blur. 

    Args:
        image (np.ndarray): The image to be preprocessed.
    Returns:
        np.ndarray. Preprocessed Image.

    Depending on the specific use case this method should be overwritten to
    use the desired preprocessing pipeline.
    """
    image_uint = to_uint8(image, mode=mode)
    image_resized = cv2.resize(image_uint, (0,0), fx=fx, fy=fy)
    image_gblur = cv2.GaussianBlur(image_resized, kernel, sigma)
    return image_gblur

def threshold_image(image, binary=True, mode="top", factor=3, **kwargs):
    """
    Thresholds the image according to one of the modes described below.

    mean:
    	Set the threshold line to be image.mean + image.std*factor.
    top:
    	Sets the threshold line to be image.max - image.std*factor, leaving just
    	the highest intensity pixels.
    bottom:
    	Sets the threshold line to be image.min + image.std*factor, removing
    	just the lowest intensity pixels
    adaptive:
    	Sets threshold line according to a weighed sum of neughborhood values
    	using a gaussian window. See 'Adaptive Thresholding' in the following
    	link for more details.
		http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    otsu:
		Sets the threshold to be between the histogram peaks of a bimodal image.
    	See "Otsu's Binarization" in the following for more details.
		http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html    	
    """
    valid_modes = set('mean', 'top', 'bottom', 'adaptive', 'otsu')
    if binary:
        th_type = cv2.THRESH_BINARY
    else:
        th_type = cv2.cv2.THRESH_TOZERO

    if mode.lower() not in valid_modes:
        error_str = "Invalid mode passed for thresholding."
        logger.error(error_str, stack_info=True)
        raise ValueError(error_str)
    elif mode.lower() == 'mean':
        _, th = cv2.threshold(image, image.mean() - image.std()*factor, 255,
                              th_type)        
    elif mode.lower() == 'top':
        _, th = cv2.threshold(image, image.max() - image.std()*factor, 255,
                              th_type)
    elif mode.lower() == 'bottom':
        _, th = cv2.threshold(image, image.min() - image.std()*factor, 255,
                              th_type)
    elif mode.lower() == "adaptive":
        blockSize = kwargs.pop("BlockSize", 11)
        C = kwargs.pop("C", 2)        
        th = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   th_type, blockSize, C)
    elif mode.lower() == "otsu":
        _, th = cv2.threshold(image, 0, 255, th_type+cv2.THRESH_OTSU)

    return th
        
                    
