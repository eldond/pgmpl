#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import os


def printd(*args, **kw):
    """
    Prints only if debug flag is turned on (greater than level)
    :param args: Things to print
    :param level: int
        Debugging level
    """
    debug = os.environ.get('PYQTMPL_DEBUG', "0")
    if int(debug) >= kw.pop('level', 1):
        print(*args)
