#!/usr/bin/env python
# -*- coding: utf-8 -*-
############
# Standard #
############
import logging

logger=logging.getLogger(__name__)

def collector(field, output):
    """
    Reimplement bluesky.callbacks.collector to not raise exception when field
    is missing. Instead, log a warning.
    """
    def f(name, event):
        try:
            output.append(event['data'][field])
            logger.debug("%s collector has collected, all output: %s",
                         field, output)
        except KeyError:
            logger.warning("did not find %s in event doc, skipping", field)

    return f
