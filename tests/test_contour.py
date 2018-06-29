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
from pgmpl.pyplot import subplots
from pgmpl.contour import QuadContourSet, ContourSet


class TestPgmplContour(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    x = np.linspace(0, 1.8, 30)
    y = np.linspace(0, 2.1, 25)
    z = (x[:, np.newaxis] - 0.94)**2 + (y[np.newaxis, :] - 1.2)**2 + 1.145
    levels = [1, 1.5, 2, 2.5, 3]
    nlvl = len(levels) * 4

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_contour(self):
        fig, axs = subplots(4, 2)
        axs[0, 0].set_title('z')
        axs[0, 0].contour(self.z)
        axs[0, 1].set_title('-z')
        axs[0, 1].contour(-self.z)

        axs[1, 0].set_title('z, levels')
        axs[1, 0].contour(self.z, self.levels)
        axs[1, 1].set_title('z, nlvl')
        axs[1, 1].contour(self.z, self.nlvl)

        axs[2, 0].set_title('x, y, z')
        axs[2, 0].contour(self.x, self.y, self.z)
        axs[2, 0].set_title('x, y, -z')
        axs[2, 1].contour(self.x, self.y, -self.z)

        axs[3, 0].set_title('x, y, z, levels')
        axs[3, 0].contour(self.x, self.y, self.z, self.levels)
        axs[3, 1].set_title('x, y, z, nlvl')
        axs[3, 1].contour(self.x, self.y, self.z, self.nlvl)

    def test_contourf(self):
        ax = Axes()
        ax.contourf(self.z)

    def test_contour_errors(self):
        ax = Axes()
        self.assertRaises(TypeError, ax.contour, self.x, self.y, self.z, self.levels, self.nlvl)

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
