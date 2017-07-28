############
# Standard #
############
import os
import logging

###############
# Third Party #
###############
import pytest
import numpy as np
from pcdsdevices.sim.areadetector.detectors import SimDetector

##########
# Module #
##########
from psbeam.images.testing import (beam_image_01, beam_image_02, beam_image_03,
                                   beam_image_04)

beam_images = [beam_image_01, beam_image_02, beam_image_03, beam_image_04]

def return_beam_image(detector):
    np.random.seed()
    idx = np.random.randint(0, len(beam_images))
    detector._n_image = idx        
    if idx == 3:
        detector._has_beam = False
    else:
        detector._has_beam = True
    return beam_images[idx].astype(np.uint8)

@pytest.fixture(scope='function')
def sim_det():
    det = SimDetector("PSB:SIM")
    # Initialize has_beam and n_image attribute
    det._n_image = -1
    det._has_beam = False
    # Spoof image2
    det.image._image = lambda : return_beam_image(det)
    return det

@pytest.fixture(scope='function')
def RE():
    """
    Standard logging runengine
    """
    RE = RunEngine({})
    return RE
