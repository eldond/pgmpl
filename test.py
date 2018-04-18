#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Tests key features of pyqtmpl
"""

# Basic imports
from __future__ import print_function, division
import unittest
import sys
import warnings

# Calculation imports
import numpy as np

# Plotting imports
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from matplotlib.colors import to_rgba

# pyqtmpl
from translate import style_translator, color_translator, symbol_translator, setup_pen_kw, plotkw_translator
from pyplot import figure, subplots
from util import printd
from examples import demo_plot


class TestPyQtMpl(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest test
    """

    verbose = False

    plot_kw_tests = [
        {'color': 'r'},
        {'color': 'r', 'alpha': 0.5},
        {'color': 'b', 'linestyle': '--', 'marker': 'x'},
        {'color': 'g', 'linestyle': ':', 'marker': 'o'},
        {'color': 'm', 'linestyle': ' ', 'marker': 'd', 'mew': 2},
    ]

    nt = len(plot_kw_tests)

    if verbose:
        print('-' * 79)
        print('\nTestPyQtMpl has {} test sets of plot keywords ready to go!\n'.format(nt))
        print('-' * 79)

    def test_color_translator(self):
        newc = [None] * self.nt
        for i in range(self.nt):
            newc[i] = color_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New colors:', newc)

    def test_style_translator(self):
        news = [None] * self.nt
        for i in range(self.nt):
            news[i] = style_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New styles:', news)
            print('QtCore.Qt.DashDotLine = {}'.format(QtCore.Qt.DashDotLine))
            print('style_translator(linestyle="-.") = {}'.format(style_translator(linestyle="-.")))
        assert style_translator(linestyle="-.") == QtCore.Qt.DashDotLine

    def test_symbol_translator(self):
        news = [None] * self.nt
        for i in range(self.nt):
            news[i] = symbol_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New symbols:', news)

    def test_setup_pen_kw(self):
        newp = [None] * self.nt
        for i in range(self.nt):
            newp[i] = setup_pen_kw(**self.plot_kw_tests[i])
        if self.verbose:
            print('New pens:', newp)

    def test_plotkw_translator(self):
        newk = [{}] * self.nt
        for i in range(self.nt):
            newk[i] = plotkw_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New keyword dictionaries:', newk)

    def test_figure(self):
        fig = figure()
        if self.verbose:
            print('test_figure: fig = {}'.format(fig))

    def test_subplots(self):
        x = np.linspace(0, 1.2, 20)
        y = x**2 + 1
        fig, axs = subplots(3, 2, sharex='all', sharey='all')
        axs[1, 1].plot(x, y)

    def test_printd(self):
        crazy_string = 'blargh hahah ralphing zerg'
        printd(crazy_string)

    def test_demo_plot(self):
        demo_plot()


if __name__ == '__main__':
    unittest.main()
