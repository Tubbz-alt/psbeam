"""
Init file for images that can be used as templates or stock images in psbeam.

To add a new directory, copy this init file into the new one, and change the
relative import dir according to how deep it is in the directory tree.
"""
############
# Standard #
############
import os

###############
# Third Party #
###############
import cv2

##########
# Module #
##########
from ..utils import get_images_from_dir   #Change this to match depth

# Grab all the images in this directory
template_images = get_images_from_dir(
    os.path.dirname(os.path.abspath(__file__)), out_type=dict)

# Make the images available in the namespace as nd.arrays and empty the dict
for name, image in template_images.items():
    exec(name + "=image")

# Cleanup
try:
    del(name, image, template_images)
except NameError:
    # Happens if there is no image in the directory
    pass

