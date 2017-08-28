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


