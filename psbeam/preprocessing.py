# Preprocessing functions used in psbeam

import cv2
import numpy as np

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
