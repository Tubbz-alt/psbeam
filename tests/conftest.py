############
# Standard #
############
import os
import logging

###############
# Third Party #
###############
import pytest
from bluesky import RunEngine
from bluesky.tests.utils import MsgCollector

logfile = os.path.join(os.path.dirname(__file__), "log.txt")
logging.basicConfig(level=logging.DEBUG, filename=logfile,
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
