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

    rgb2d = np.zeros((8, 8, 3))
    rgb2d[0, 0, :] = 0.9
    rgb2d[4, 4, :] = 1
    rgb2d[3, 2, 0] = 0.5
    rgb2d[2, 3, 1] = 0.7
    rgb2d[3, 3, 2] = 0.6

    x1 = x
    x2 = np.linspace(0, 2.1, 25)
    two_d_data = (x1[:, np.newaxis] - 0.94)**2 + (x2[np.newaxis, :] - 1.2)**2

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
        from pgmpl.axes import AxesImage
        a = self.rgb2d
        ax = Axes()
        img = ax.imshow(a)
        ax1 = Axes()
        img1 = ax1.imshow(a[:, :, 0:2])
        ax2 = Axes()
        img2 = ax2.imshow(self.two_d_data)
        assert isinstance(img, AxesImage)
        assert isinstance(img1, AxesImage)
        assert isinstance(img2, AxesImage)
        if self.verbose:
            print('test_axes_imshow: ax = {}, ax1 = {}, ax2 = {}, img = {}, img1 = {}, img2 = {}'.format(
                ax, ax1, ax2, img, img1, img2))

    def test_axes_contour(self):
        a = sum(self.rgb2d, 2) * 10
        levels = [0, 0.5, 1.2, 5, 9, 10, 20, 30]
        print('shape(a) = {}'.format(np.shape(a)))
        ax = Axes()
        ax1 = Axes()
        ax.contour(a)
        ax1.contourf(a)
        if self.verbose:
            print('test_axes_contour: ax = {}, contours = {}, ax1 = {}, contourfs = {}'.format(
                ax, contours, ax1, contourfs))

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
