############
# Standard #
############
import os
import logging
from pathlib import Path

###############
# Third Party #
###############
import pytest
import numpy as np
from bluesky import RunEngine
from bluesky.tests.utils import MsgCollector
from pcdsdevices.sim.areadetector.detectors import SimDetector

##########
# Module #
##########
from psbeam.images.testing import (beam_image_01, beam_image_02, beam_image_03,
                                   beam_image_04)
from psbeam.images.testing.hx2 import images as images_hx2
from psbeam.images.testing.dg3 import images as images_dg3

beam_images = [beam_image_01, beam_image_02, beam_image_03, beam_image_04]

#################
# Logging Setup #
#################

#Default logfile
logfile = Path(os.path.dirname(__file__)) / "../logs/log.txt"
if not logfile.parent.exists():
    logfile.parent.mkdir(parents=True)
    
#Enable the logging level to be set from the command line
def pytest_addoption(parser):
    parser.addoption("--log", action="store", default="DEBUG",
                     help="Set the level of the log")
    parser.addoption("--logfile", action="store", default=str(logfile),
                     help="Write the log output to specified file path")

#Create a fixture to automatically instantiate logging setup
@pytest.fixture(scope='session', autouse=True)
def set_level(pytestconfig):
    #Read user input logging level
    log_level = getattr(logging, pytestconfig.getoption('--log'), None)

    #Report invalid logging level
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level : {}".format(log_level))

    #Create basic configuration
    logging.basicConfig(level=log_level,
                        filename=pytestconfig.getoption('--logfile'),
                        format='%(asctime)s - %(levelname)s ' +
                               '- %(name)s - %(message)s')

logger = logging.getLogger(__name__)
logger.info("pytest start")
run_engine_logger = logging.getLogger("RunEngine")


@pytest.fixture(scope='function')
def RE():
    """
    Standard logging runengine
    """
    RE = RunEngine({})
    collector = MsgCollector(msg_hook=run_engine_logger.debug)
    RE.msg_hook = collector
    return RE

def yield_seq_beam_image(detector, images, idx=0):
    while True:
        val = idx % len(images)
        yield images[val].astype(np.uint8)
        idx += 1        

def _next_image(det, gen):
    det.image.array_counter.value += 1
    return next(gen)
        
@pytest.fixture(scope='function')
def sim_det_01():
    det = SimDetector("PSB:SIM:01")
    # Spoof image
    yield_image = yield_seq_beam_image(det, beam_images, idx=0)
    det.image._image = lambda : _next_image(det, yield_image)
    return det

@pytest.fixture(scope='function')
def sim_det_02():
    det = SimDetector("PSB:SIM:02")
    # Spoof image
    yield_image = yield_seq_beam_image(det, beam_images, idx=0)
    det.image._image = lambda : _next_image(det, yield_image)
    return det

@pytest.fixture(scope='function')
def sim_hx2():
    det = SimDetector("PSB:SIM:HX2")
    # Spoof image
    yield_image = yield_seq_beam_image(det, images_hx2, idx=0)
    det.image._image = lambda : _next_image(det, yield_image)
    return det

@pytest.fixture(scope='function')
def sim_dg3():
    det = SimDetector("PSB:SIM:DG3")
    # Spoof image
    yield_image = yield_seq_beam_image(det, images_dg3, idx=0)
    det.image._image = lambda : _next_image(det, yield_image)
    return det


