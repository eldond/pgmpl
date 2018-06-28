#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for pyplot.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.pyplot import *


class TestPgmplPyplot(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    x = np.linspace(0, 1.2, 20)
    y = x**2 + 1

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_pyplot_figure(self):
        fig = figure()
        fig.close()

    def test_pyplot_subplots(self):
        fig, axs = subplots(3, 2, sharex='all', sharey='all')
        axs[1, 1].plot(self.x, self.y)
        fig.close()

    def test_pyplot_axes(self):
        ax = axes()
        ax.plot(self.x, self.y)
        close()

    def test_pyplot_gcf(self):
        fig = gcf()
        assert isinstance(fig, Figure)
        fig.close()

    def test_pyplot_gca(self):
        ax = gca()
        assert isinstance(ax, Axes)
        close()

    def test_pyplot_close(self):
        fig = gcf()
        close(fig)
        close()  # close should do its own gcf if it doesn't get a fig argument

    def test_pyplot_plot(self):
        plot(self.x, self.y)
        close()

    def test_pyplot_subplots_adjust(self):
        plot(self.x, self.y**2+self.x**2+250)
        subplots_adjust(left=0.5, right=0.99)
        close()

    def test_pyplot_text_functions(self):
        suptitle('super title text')
        close()

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
