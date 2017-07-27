############
# Standard #
############
import logging

###############
# Third Party #
###############
import cv2
import pytest
from pcdsdevices.sim import pim
from bluesky.plans import run_wrapper

##########
# Module #
##########
from .utils import collector
# from psbeam.plans import measure_beam_quality

# logger = logging.getLogger(__name__)

# def test_measure_beam_quality(RE):
#     # import ipdb; ipdb.set_trace()
#     # import IPython; IPython.embed()
#     p = pim.PIM("TEST")
#     test_image = cv2.imread("tests/test_yag_images_01/image97.jpg", 0)
#     p.detector.image1.image = test_image
    
#     res = {}
#     col = collector("Sim Detector", res)
    
    
#     RE(run_wrapper(measure_beam_quality(p.detector, p.detector.image1, 
#                                         read_rate=100, num_reads=10)),
#        subs={'event':[col]})

#     import IPython; IPython.embed()


                   

    
