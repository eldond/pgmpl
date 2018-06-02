#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import os
import numpy as np


def set_debug(enable=True):
    flag = str(int(bool(enable)))
    os.environ['PGMPL_DEBUG'] = flag
    printd('pgmpl debugging set to {}'.format(flag))


def printd(*args, **kw):
    """
    Prints only if debug flag is turned on (greater than level)
    :param args: Things to print
    :param level: int
        Debugging level
    """
    debug = os.environ.get('PGMPL_DEBUG', "0")
    if int(debug) >= kw.pop('level', 1):
        print(*args)


def tolist(x):
    return np.ndarray.tolist(np.atleast_1d(x))


def is_iterable(x):  # https://stackoverflow.com/a/1952481/6605826
    """Test if x is iterable"""
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def is_numeric(value):
    """
    Convenience function check if value is numeric, taken from OMFIT
    :param value: value to check
    :return: True/False
    """
    try:
        0+value
        return True
    except TypeError:
        return False

