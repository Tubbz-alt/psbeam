#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to hold all the psbeam exceptions.
"""
############
# Standard #
############
import logging

logger = logging.getLogger(__name__)

class BeamException(Exception):
    """
    Base exception class for psbeam.
    """
    def __init__(self, msg='', *args, **kwargs):
        self.msg = msg
        super().__init__(*args, **kwargs)

    def __str__(self):
        logger.error(self.msg, stack_info=True)
        return repr(self.msg)

class InputError(BeamException):
    """
    Generic exception raised if an invalid input was used.
    """
    pass

class NoContoursDetected(BeamException):
    """
    Error raised if an operation requiring contours is requested but no contours
    were returned by the get_contours function.
    """
    def __init__(self, msg='', *args, **kwargs):
        if not msg:
            msg = "Cannot perform operation; No contours found."
        super().__init__(msg=msg, *args, **kwargs)

class NoBeamDetected(BeamException):
    """
    Exception raised if an operation requiring the beam is requested but no beam
    is actually present.
    """
    def __init__(self, msg='', *args, **kwargs):
        if not msg:
            msg = "Cannot perform operation; No beam found."
        super().__init__(msg=msg, *args, **kwargs)

        
class MomentOutOfRange(BeamException):
    """
    Exception raised a beam moment is out of its designated range.
    """
    def __init__(self, msg='', *args, **kwargs):
        if not msg:
            msg = "Moment is out of specified range."
        super().__init__(msg=msg, *args, **kwargs)
    
