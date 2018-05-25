#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for figure.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.figure import Figure

class TestPgmplFigure(unittest.TestCase):

    verbose = False

    def test_figure(self):
        fig1 = Figure()
        assert isinstance(fig1, Figure)
        if self.verbose:
            print('test_figure: fig1 = {}'.format(fig1))
        fig1.close()

    def test_fig_methods(self):
        from pgmpl.axes import Axes
        fig = Figure()
        ax = fig.gca()
        assert isinstance(ax, Axes)
        fig.suptitle('suptitle text in unittest')
        ax2 = fig.gca()
        assert ax2 == ax
        fig.clear()
        fig.close()
        assert fig.clearfig == fig.clear  # Make sure this assignment didn't break.
        if self.verbose:
            print('test_fig_methods: fig = {}, ax = {}'.format(fig, ax))

    def test_fig_plot(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, 1])
        if self.verbose:
            print('test_fig_plot: fig = {}, ax = {}'.format(fig, ax))
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
        if self.verbose:
            print('test_figure_set_subplotpars: fig = {}, ax = {}, ax2 = {}'.format(fig, ax, ax2))
        fig.close()

    def test_fig_colorbar(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.linspace(0, 1, 21)
        y = np.linspace(0, 1, 6)
        aa = x[:, np.newaxis] * y[np.newaxis, :] * 2.5
        img = ax.imshow(aa)
        fig.colorbar(img)
        if self.verbose:
            print('test_fig_colorbar: ax = ax')
        fig.close()


if __name__ == '__main__':
    unittest.main()
