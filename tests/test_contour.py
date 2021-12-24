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

    x0 = 0.94
    y0 = 2.2
    x02 = x0 - 0.45
    y02 = y0 - 0.7
    z0 = 1.145
    slant = .0
    x = np.linspace(0, 1.8, 30)
    y = np.linspace(1, 3.1, 25)
    dx = x[np.newaxis, :] - x0
    dy = y[:, np.newaxis] - y0
    dx2 = x[np.newaxis, :] - x02
    dy2 = y[:, np.newaxis] - y02
    z = dx**2 + dy**2 + z0 + slant*(dx+dy)**2
    z2 = -(dx2**2 + dy2**2 + (slant + 1.2) * (dx2 + dy2)**2)
    z3 = dx**2 + dy**2 + (slant - 1.2) * (dx + dy)**2
    levels = [1.2, 1.5, 2, 2.5, 2.95]
    nlvl = len(levels) * 4

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_contour(self):
        fig, axs = subplots(4, 2)
        fig.suptitle('TestPgmplContour.test_contour')
        axs[0, 0].set_title('z')
        axs[0, 0].contour(self.z)
        axs[0, 1].set_title('-z')
        axs[0, 1].contour(-self.z, linestyles=['--', '-.', ':'])

        axs[1, 0].set_title('z, levels')
        axs[1, 0].contour(self.z, self.levels)
        axs[1, 1].set_title('z, nlvl')
        axs[1, 1].contour(self.z, self.nlvl, linewidths=[3, 2, 1])

        axs[2, 0].set_title('x, y, z')
        axs[2, 0].contour(self.x, self.y, self.z)
        axs[2, 0].set_title('x, y, -z')
        axs[2, 1].contour(self.x, self.y, -self.z, colors=['r', 'g', 'b'])

        axs[3, 0].set_title('x, y, z, levels')
        axs[3, 0].contour(self.x, self.y, self.z, self.levels)
        axs[3, 1].set_title('x, y, z, nlvl')
        axs[3, 1].contour(self.x, self.y, self.z, self.nlvl, linestyles=['-', '--', '-.', ':'])

        # Repeat with matplotlib to check whether the same figure is produced
        import matplotlib
        from matplotlib import pyplot
        fig, axs = pyplot.subplots(4, 2)
        fig.suptitle('TestPgmplContour.test_contour')
        axs[0, 0].set_title('z')
        axs[0, 0].contour(self.z)
        axs[0, 1].set_title('-z')
        axs[0, 1].contour(-self.z, linestyles=['--', '-.', ':'])

        axs[1, 0].set_title('z, levels')
        axs[1, 0].contour(self.z, self.levels)
        axs[1, 1].set_title('z, nlvl')
        axs[1, 1].contour(self.z, self.nlvl, linewidths=[3, 2, 1])

        axs[2, 0].set_title('x, y, z')
        axs[2, 0].contour(self.x, self.y, self.z)
        axs[2, 0].set_title('x, y, -z')
        axs[2, 1].contour(self.x, self.y, -self.z, colors=['r', 'g', 'b'])

        axs[3, 0].set_title('x, y, z, levels')
        axs[3, 0].contour(self.x, self.y, self.z, self.levels)
        axs[3, 1].set_title('x, y, z, nlvl')
        axs[3, 1].contour(self.x, self.y, self.z, self.nlvl, linestyles=['-', '--', '-.', ':'])

        # # Uncomment these to see figures during manual testing and development:
        # pyplot.show()
        # import pgmpl
        # pgmpl.app.exec_()

    def test_contourf(self):
        fig, axs = subplots(4, 2)
        fig.suptitle('TestPgmplContour.test_contourf')
        axs[0, 0].set_title('z')
        axs[0, 0].contourf(self.z)

        axs[0, 1].set_title('z, nlvl')
        axs[0, 1].contourf(self.z, self.nlvl)

        axs[1, 0].set_title('z2')
        axs[1, 0].contourf(self.z2)

        axs[1, 1].set_title('z3')
        axs[1, 1].contourf(self.z3)

        axs[2, 0].set_title('x, y, z')
        axs[2, 0].contourf(self.x, self.y, self.z)

        axs[2, 1].set_title('x, y, z, nlvl')
        axs[2, 1].contourf(self.x, self.y, self.z, self.nlvl)

        axs[3, 0].set_title('x, y, z2')
        axs[3, 0].contourf(self.x, self.y, self.z2)

        axs[3, 1].set_title('x, y, z3')
        axs[3, 1].contourf(self.x, self.y, self.z3)

        # Repeat with matplotlib to check whether the same figure is produced
        import matplotlib
        from matplotlib import pyplot
        figm, axsm = pyplot.subplots(4, 2)
        figm.suptitle('TestPgmplContour.test_contourf')
        axsm[0, 0].set_title('z')
        axsm[0, 0].contourf(self.z)

        axsm[0, 1].set_title('z, nlvl')
        axsm[0, 1].contourf(self.z, self.nlvl)

        axsm[1, 0].set_title('z2')
        axsm[1, 0].contourf(self.z2)

        axsm[1, 1].set_title('z3')
        axsm[1, 1].contourf(self.z3)

        axsm[2, 0].set_title('x, y, z')
        axsm[2, 0].contourf(self.x, self.y, self.z)

        axsm[2, 1].set_title('x, y, z, nlvl')
        axsm[2, 1].contourf(self.x, self.y, self.z, self.nlvl)

        axsm[3, 0].set_title('x, y, z2')
        axsm[3, 0].contourf(self.x, self.y, self.z2)

        axsm[3, 1].set_title('x, y, z3')
        axsm[3, 1].contourf(self.x, self.y, self.z3)

        # Uncomment these to see figures during manual testing and development:
        pyplot.show()
        import pgmpl
        pgmpl.app.exec_()

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
