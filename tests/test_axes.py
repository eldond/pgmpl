#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for axes.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.axes import Axes


class TestPgmplAxes(unittest.TestCase):
    """
    Most test functions simply test one method of Axes. test_axes_plot tests Axes.plot(), for example.
    More complicated behaviors will be mentioned in function docstrings.
    """

    verbose = False

    x = np.linspace(0, 1.8, 30)
    y = x**2 + 2.5
    z = x**3 - x**2 * 1.444

    def test_axes_init(self):
        ax = Axes()
        if self.verbose:
            print('test_axes_init: ax = {}'.format(ax))

    def test_axes_plot(self):
        ax = Axes()
        ax.plot(self.x, self.y, color='r')
        if self.verbose:
            print('test_axes_plot: ax = {}'.format(ax))

    def test_axes_scatter(self):
        ax = Axes()
        ax.scatter(self.x, self.y, c=self.z)
        ax.scatter(self.x, self.y, c='b')
        ax.scatter(self.x, self.y, c=self.z, cmap='plasma', marker='s', linewidths=1, edgecolors='r')
        # noinspection PyTypeChecker
        ax.scatter(self.x, self.x*0, c=self.x, cmap='jet', marker=None,
                   verts=[(0, 0), (0.5, 0.5), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, -0.5)])

    def test_axes_imshow(self):
        a = np.zeros((8, 8, 3))
        a[0, 0, :] = 0.9
        a[4, 4, :] = 1
        a[3, 2, 0] = 0.5
        a[2, 3, 1] = 0.7
        a[3, 3, 2] = 0.6
        ax = Axes()
        ax.imshow(a)
        ax1 = Axes()
        ax1.imshow(a[:, :, 0:2])
        if self.verbose:
            print('test_axes_imshow: ax = {}, ax1 = {}'.format(ax, ax1))

    def test_axes_err(self):
        ax = Axes()
        yerr = self.y*0.1
        ax.errorbar(self.x, self.y, yerr, color='r')
        ax.fill_between(self.x, -self.y-yerr-1, -self.y+yerr-1)
        if self.verbose:
            print('test_axes_err: ax = {}'.format(ax))

    def test_axes_lines(self):
        ax = Axes()
        ax.axhline(0.5, linestyle='--', color='k')
        ax.axvline(0.5)
        ax.axvline(0.75, linestyle='-', color='b')
        if self.verbose:
            print('test_axes_lines: ax = {}'.format(ax))

    def test_axes_xyaxes(self):
        ax = Axes()
        ax.plot([0, 1], [1, 2])
        ax.set_ylabel('ylabel')
        ax.set_xlabel('xlabel')
        ax.set_title('title title title')
        ax.set_xlim([-1, 2])
        ax.set_ylim([-2, 4])
        ax.set_xscale('linear')
        ax.set_yscale('log')
        if self.verbose:
            print('test_axes_xyaxes: ax = {}'.format(ax))

    def test_axes_aspect(self):
        ax = Axes()
        ax.plot([0, 10, 0, 1])
        ax.set_aspect('equal')
        if self.verbose:
            print('test_axes_aspect: ax = {}'.format(ax))

    def test_axes_clear(self):
        ax = Axes()
        ax.plot(self.y, self.x)  # Switch them so the test doesn't get bored.
        ax.clear()
        # Should add something to try to get the number of objects on the test and assert that there are none
        if self.verbose:
            print('test_axes_clear: ax = {}'.format(ax))

    def test_Legend(self):
        """Tests both the legend method of Axes and the Legend class implicitly"""
        ax = Axes()
        line = ax.plot(self.x, self.y, label='y(x) plot')
        leg = ax.legend()
        leg.addItem(line, name='yx plot')
        leg.draggable()
        leg.clear()
        if self.verbose:
            print('test_axes_Legend: ax = {}, leg = {}'.format(ax, leg))


if __name__ == '__main__':
    unittest.main()
