#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for colorbar.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.colorbar import Colorbar


class TestPgmplColorbar(unittest.TestCase):

    verbose = False

    # Make some test data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 5)
    a = x[:, np.newaxis] * y[np.newaxis, :]

    def test_colorbar(self):
        from pgmpl.pyplot import subplots
        fig, ax = subplots(1)
        img = ax.imshow(self.a)
        cb = fig.colorbar(img)
        assert isinstance(cb, Colorbar)
        if self.verbose:
            print('test_colorbar: ax = {}, cb = {}'.format(ax, cb))
        fig.close()


if __name__ == '__main__':
    unittest.main()
