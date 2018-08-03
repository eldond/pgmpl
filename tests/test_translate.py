#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for translate.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np
from pyqtgraph import QtCore, QtGui
import copy
from matplotlib import rcParams

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.util import set_debug
from pgmpl.translate import defaults_from_rcparams, color_translator, style_translator, symbol_translator, \
    setup_pen_kw, plotkw_translator, dealias, color_map_translator


class TestPgmplTranslate(unittest.TestCase):
    """
    Each test function tests one of the functions in translate.py.
    The test function names are test_ + the name of the function they test.
    """

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    plot_kw_tests = [
        {'color': 'r'},
        {'color': 'r', 'alpha': 0.5},
        {'color': 'b', 'linestyle': '--', 'marker': 'x'},
        {'color': 'g', 'linestyle': ':', 'marker': 'o'},
        {'color': 'm', 'linestyle': ' ', 'marker': 'd', 'mew': 2},
        {'color': ' ', 'markeredgewidth': 1.5},
        {'linestyle': '-.'},
        {'marker': '+', 'alpha': 0.7},
    ]

    nt = len(plot_kw_tests)

    def printv(self, *args):
        if self.verbose:
            print(*args)

    if verbose:
        print('-' * 79)
        print('\nTestPgmplTranslate has {} test sets of plot keywords ready to go!\n'.format(nt))
        print('-' * 79)

    def test_defaults_from_rcparams(self):
        unique_linewidth = 5.1234958293 + rcParams['lines.linewidth']
        ans0 = defaults_from_rcparams({})
        ans1 = defaults_from_rcparams({'linewidth': unique_linewidth})
        ans0b = copy.deepcopy(ans0)
        ans0b['linewidth'] = unique_linewidth
        assert ans1 == ans0b
        assert ans1 != ans0

    def test_color_translator(self):
        newc = [None] * self.nt
        for i in range(self.nt):
            newc[i] = color_translator(**self.plot_kw_tests[i])
        assert all(color_translator(color='r') == np.array([255., 0., 0., 255.]))
        assert all(color_translator(color='g') == np.array([0., 255./2., 0., 255.]))
        assert all(color_translator(color='b') == np.array([0., 0., 255., 255.]))
        assert all(color_translator(color='k', alpha=0.5) == np.array([0., 0., 0., 255./2.]))
        assert all(color_translator(color=(1, 0.5, 1)) == np.array([255., 255/2., 255., 255.]))
        assert all(color_translator(color=(1, 0.5, 1, 0.5)) == np.array([255., 255/2., 255., 255./2.]))
        assert all(color_translator(color=(1, 0.5, 1), alpha=0.5) == np.array([255., 255/2., 255., 255./2.]))

    def test_style_translator(self):
        news = [None] * self.nt
        for i in range(self.nt):
            news[i] = style_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New styles:', news)
            print('QtCore.Qt.DashDotLine = {}'.format(QtCore.Qt.DashDotLine))
            print('style_translator(linestyle="-.") = {}'.format(style_translator(linestyle="-.")))
        assert style_translator(linestyle="-.") == QtCore.Qt.DashDotLine
        assert style_translator(linestyle="--") == QtCore.Qt.DashLine
        assert style_translator(linestyle=":") == QtCore.Qt.DotLine
        assert style_translator(linestyle="-") == QtCore.Qt.SolidLine

    def test_symbol_translator(self):
        news = [None] * self.nt
        for i in range(self.nt):
            news[i] = symbol_translator(**self.plot_kw_tests[i])
        assert symbol_translator(marker='o') == 'o'
        assert symbol_translator(marker='+') == '+'
        assert symbol_translator(marker='v') == 't'
        assert symbol_translator(marker='^') == 't1'
        custom_markers = '_x|,.'
        for custom in custom_markers:
            assert isinstance(symbol_translator(marker=custom), QtGui.QPainterPath)

    def test_setup_pen_kw(self):
        newp = [None] * self.nt
        for i in range(self.nt):
            newp[i] = setup_pen_kw(**self.plot_kw_tests[i])
            assert isinstance(newp[i], QtGui.QPen)

    def test_plotkw_translator(self):
        newk = [{}] * self.nt
        for i in range(self.nt):
            newk[i] = plotkw_translator(**self.plot_kw_tests[i])

    def test_dealias(self):
        test_dict = {'lw': 5, 'ls': '--', 'mec': 'r', 'markeredgewidth': 1, 'blah': 0}
        correct_answer = {'linewidth': 5, 'linestyle': '--', 'markeredgecolor': 'r', 'markeredgewidth': 1, 'blah': 0}
        test_answer = dealias(**test_dict)
        assert correct_answer == test_answer  # https://stackoverflow.com/a/5635309/6605826

        assert dealias(lw=8) == {'linewidth': 8}
        assert dealias(blah=58) == {'blah': 58}
        assert dealias(mec='r') == {'markeredgecolor': 'r'}

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

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
