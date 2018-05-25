#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for pyplot.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.pyplot import *


class TestPgmplPyplot(unittest.TestCase):

    verbose = False

    x = np.linspace(0, 1.2, 20)
    y = x**2 + 1

    def test_pyplot_figure(self):
        fig = figure()
        if self.verbose:
            print('test_pyplot_figure: fig = {}'.format(fig))
        fig.close()

    def test_pyplot_subplots(self):
        fig, axs = subplots(3, 2, sharex='all', sharey='all')
        axs[1, 1].plot(self.x, self.y)
        if self.verbose:
            print('test_pyplot_subplots: axs = {}'.format(axs))
        fig.close()

    def test_pyplot_axes(self):
        ax = axes()
        ax.plot(self.x, self.y)
        if self.verbose:
            print('test_pyplot_axes: ax = {}'.format(ax))
        close()

    def test_pyplot_gcf(self):
        fig = gcf()
        assert isinstance(fig, Figure)
        if self.verbose:
            print('test_pyplot_gcf: fig = {}'.format(fig))
        fig.close()

    def test_pyplot_gca(self):
        ax = gca()
        assert isinstance(ax, Axes)
        if self.verbose:
            print('test_pyplot_gca: ax = {}'.format(ax))
        close()

    def test_pyplot_close(self):
        fig = gcf()
        close(fig)
        if self.verbose:
            print('test_pyplot_close: fig = {}'.format(fig))
        close()  # close should do its own gcf if it doesn't get a fig argument

    def test_pyplot_plot(self):
        plot(self.x, self.y)
        if self.verbose:
            print('test_pyplot_plot: done')
        close()

    def test_pyplot_subplots_adjust(self):
        plot(self.x, self.y**2+self.x**2+250)
        subplots_adjust(left=0.5, right=0.99)
        if self.verbose:
            print('test_pyplot_subplots_adjust: done')
        close()

    def test_pyplot_text_functions(self):
        suptitle('super title text')
        if self.verbose:
            print('test_pyplot_text_functions: done')
        close()


if __name__ == '__main__':
    unittest.main()
