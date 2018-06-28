#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for figure.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.figure import Figure


class TestPgmplFigure(unittest.TestCase):
    """
    Most test functions simply test one method of Figure. test_fig_colorbar tests Figure.colorbar(), for example.
    More complicated behaviors will be mentioned in function docstrings.
    """

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_figure(self):
        fig1 = Figure()
        assert isinstance(fig1, Figure)
        fig1.close()

    def test_fig_methods(self):
        """Test Figure methods gca, show, clear, and close"""
        from pgmpl.axes import Axes
        fig = Figure()
        ax = fig.gca()
        assert isinstance(ax, Axes)
        fig.suptitle('suptitle text in unittest')
        ax2 = fig.gca()
        assert ax2 == ax
        fig.show()
        fig.clear()
        fig.close()
        assert fig.clearfig == fig.clear  # Make sure this assignment didn't break.

    def test_fig_add_subplot(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, 1])
        fig.close()

    def test_figure_set_subplotpars(self):
        from matplotlib.figure import SubplotParams
        sp = SubplotParams(left=0.5, right=0.99, bottom=0.01, top=0.5, wspace=0.2, hspace=0.2)
        fig = Figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.plot([0, 1, 0, 1, 0, 1])
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot([1, 0, 1])
        fig.set_subplotpars(sp)
        fig.close()

    def test_fig_colorbar(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 6)
        aa = x[:, np.newaxis] * y[np.newaxis, :] * 2.5
        img = ax.imshow(aa)
        fig.colorbar(img)
        fig.close()

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
