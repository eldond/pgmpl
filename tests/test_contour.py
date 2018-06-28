#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for contour.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np
import warnings

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.axes import Axes
from pgmpl.contour import QuadContourSet, ContourSet


class TestPgmplContour(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))


if __name__ == '__main__':
    unittest.main()
