#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for translate.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np
from PyQt4 import QtCore
import copy
from matplotlib import rcParams

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.translate import defaults_from_rcparams, color_translator, style_translator, symbol_translator, \
    setup_pen_kw, plotkw_translator, dealias, color_map_translator


class TestPgmplTranslate(unittest.TestCase):

    verbose = False

    plot_kw_tests = [
        {'color': 'r'},
        {'color': 'r', 'alpha': 0.5},
        {'color': 'b', 'linestyle': '--', 'marker': 'x'},
        {'color': 'g', 'linestyle': ':', 'marker': 'o'},
        {'color': 'm', 'linestyle': ' ', 'marker': 'd', 'mew': 2},
        {'color': ' ', 'markeredgewidth': 1.5},
    ]

    nt = len(plot_kw_tests)

    if verbose:
        print('-' * 79)
        print('\nTestPgmpl has {} test sets of plot keywords ready to go!\n'.format(nt))
        print('-' * 79)
        set_debug(True)

    def test_defaults_from_rcparams(self):
        unique_linewidth = 5.1234958293
        ans0 = defaults_from_rcparams({})
        ans1 = defaults_from_rcparams({'linewidth': unique_linewidth})
        ans0b = copy.deepcopy(ans0)
        ans0b['linewidth'] = unique_linewidth
        assert ans1 == ans0b
        if rcParams['lines.linewidth'] == unique_linewidth:
            assert ans1 == ans0
        else:
            assert ans1 != ans0

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

    def test_dealias(self):
        test_dict = {'lw': 5, 'ls': '--', 'mec': 'r', 'markeredgewidth': 1, 'blah': 0}
        correct_answer = {'linewidth': 5, 'linestyle': '--', 'markeredgecolor': 'r', 'markeredgewidth': 1, 'blah': 0}
        test_answer = dealias(**test_dict)
        assert correct_answer == test_answer  # https://stackoverflow.com/a/5635309/6605826
        if self.verbose:
            print('test_dealias: test_answer = {}'.format(test_answer))

    def test_color_map_translator(self):
        x = [0, 1, 2, 3, 5, 9, 10, 22]
        m1 = color_map_translator(1.579, cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256, alpha=0.5)
        m2 = color_map_translator([1, 2, 3], cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256)
        m3 = color_map_translator(x, cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256)
        m4 = color_map_translator(x, cmap='plasma', norm=None, vmin=None, vmax=None, clip=False, ncol=256)
        assert len(m1) == 1
        assert len(m2) == 3
        assert len(m3) == len(x)
        assert any((np.atleast_1d(m3) != np.atleast_1d(m4)).flatten())
        if self.verbose:
            print('test_color_map_translator: m4 = {}'.format(m4))


if __name__ == '__main__':
    unittest.main()
