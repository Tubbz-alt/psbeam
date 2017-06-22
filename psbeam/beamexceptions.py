#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to hold all the psbeam exceptions.
"""
############
# Standard #
############
import logging


class BeamException(Exception):
    """
    Base exception class for psbeam.
    """
    pass


class NoBeamPresent(BeamException):
    """
    Exception raised if an operation requiring the beam is requested but no beam
    is actually present.
    """
    def __init__(self, *args, **kwargs):
        self.msg = kwargs.pop("msg", "Cannot perform operation; No beam found.")
        super().__init__(*args, **kwargs)
    def __str__(self):
        return repr(self.msg)
