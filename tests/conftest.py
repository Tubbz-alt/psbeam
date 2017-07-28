############
# Standard #
############
import os
import logging

###############
# Third Party #
###############
import pytest
from pcdsdevices.sim.pim import PIM

@pytest.fixture(scope='function')
def RE():
    """
    Standard logging runengine
    """
    RE = RunEngine({})
    return RE

@pytest.fixture(scope='function')
def fake_pim():
    p = PIM("TEST")
    
