#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import os


def printd(*args):
    """
    Prints only if debug flag is turned on
    :param args: Things to print
    """
    debug = os.environ.get('PYQTMPL_DEBUG', "0")
    if debug != "0":
        print(*args)
