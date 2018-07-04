#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for axes.py
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


class TestPgmplAxes(unittest.TestCase):
    """
    Most test functions simply test one method of Axes. test_axes_plot tests Axes.plot(), for example.
    More complicated behaviors will be mentioned in function docstrings.
    """

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

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

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_axes_plot(self):
        ax = Axes()
        ax.plot(self.x, self.y, color='r')
        for i in range(len(ax.prop_cycle)+1):  # Use up all the colors in the prop cycle so it has to loop
            ax.plot(self.x, self.y+i)

    def test_axes_scatter(self):
        ax = Axes()
        ax.scatter(self.x, -self.y)
        ax.scatter(self.x, self.y, c=self.z)
        ax.scatter(self.x, self.y, c='b', aspect='equal')
        ax.scatter(self.x, self.y, c=self.z, cmap='plasma', marker='s', linewidths=1, edgecolors='r')
        # noinspection PyTypeChecker
        ax.scatter(self.x, self.x*0, c=self.x, cmap='jet', marker=None,
                   verts=[(0, 0), (0.5, 0.5), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, -0.5)])
        ax.scatter(data={'x': self.x, 'y': self.y, 'c': self.z, 's': 10})

    def test_axes_imshow(self):
        from pgmpl.axes import AxesImage
        a = self.rgb2d
        ax = Axes()
        img = ax.imshow(a)
        ax1 = Axes()
        img1 = ax1.imshow(data={'x': a[:, :, 0:2]}, aspect='equal')
        ax2 = Axes()
        img2 = ax2.imshow(self.two_d_data)
        assert isinstance(img, AxesImage)
        assert isinstance(img1, AxesImage)
        assert isinstance(img2, AxesImage)
        ax.imshow(data={'x': a})
        self.printv('      test_axes_imshow: ax = {}, ax1 = {}, ax2 = {}, img = {}, img1 = {}, img2 = {}'.format(
               ax, ax1, ax2, img, img1, img2))

    def test_axes_contour(self):
        a = sum(self.rgb2d, 2) * 10
        levels = [0, 0.5, 1.2, 5, 9, 10, 20, 30]
        ax = Axes()
        ax.contour(a)
        ax1 = Axes()
        ax1.contourf(a)

        ax2 = Axes()
        ax2.contour(a, levels)
        ax3 = Axes()
        ax3.contour(a, 3)

    def test_axes_imshow_warnings(self):
        from pgmpl.axes import AxesImage
        a = self.rgb2d
        ax = Axes()

        warnings_expected = 8
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warnings.
            img = ax.imshow(
                data={'x': a, 'unrecognized': 'thingy'}, shape=np.shape(a), imlim=55, interpolation='nearest',
                filternorm=2, filterrad=5.0, resample=True, url='google.com', blah=True)  # 8 warnings
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        assert isinstance(img, AxesImage)  # It should still return the instance using the implemented keywords.
        self.printv('      test_axes_imshow_warnings: tried to call Axes.imshow instance using unimplemented keywords '
                    'and got {}/{} warnings. img = {}'.format(len(w), warnings_expected, img))

    def test_axes_warnings(self):
        ax = Axes()
        warnings_expected = 9
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warnings.
            ax.text(0.5, 0.91, 'text1', withdash=True)  # 1 warning
            ax.fill_between(self.x, self.y*0.1, self.y*1.1, step='pre')  # 1 warning
            ax.fill_between(self.x, -self.y - 10, self.y - 10,
                            where=(self.x >= np.mean(self.x)), interpolate=True)  # 1 warning
            ax.set_xlim([-1, 2], emit=False, auto=True, blah=True)  # 3 warnings
            ax.set_ylim([-1, 2], emit=False, auto=True, blah=True)  # 3 warnings
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        self.printv('      test_axes_warnings: tried to make probelmatic calls to Axes methods '
                    'and got {}/{} warnings. ax = {}'.format(len(w), warnings_expected, ax))

    def test_axes_err(self):
        ax = Axes()
        yerr = self.y*0.1
        ax.errorbar(-self.x, self.y, yerr, color='r')
        ax.errorbar(self.x, self.y, yerr, color='r', capsize=6, capthick=1.25, marker='s', ecolor='m')
        ax.errorbar(data={'x': -self.x, 'y': -self.y, 'yerr': yerr})
        ax.errorbar(self.x, self.y, yerr, elinewidth=2.5, markeredgewidth=1.2, barsabove=True)
        ax.errorbar(self.x, self.y-20, yerr, capsize=0, capthick=0)
        ax.fill_between(self.x, -self.y-yerr-1, -self.y+yerr-1)
        ax.fill_between(data={'x': -self.x, 'y1': 10-self.y-yerr-1, 'y2': -self.y+yerr-1})
        ax.fill_between(self.x, -self.y - yerr - 1, -self.y + yerr - 1, where=(self.x >= np.mean(self.x)))
        ax.fill_between(self.x, self.y+20, 0)

    def test_axes_lines(self):
        ax = Axes()
        ax.axhline(0.5, linestyle='--', color='k')
        ax.axvline(0.5)
        ax.axvline(0.75, linestyle='-', color='b')

    def test_axes_xyaxes(self):
        ax = Axes()
        ax.plot([1, 5], [1, 2])
        ax.set_ylabel('ylabel')
        ax.set_xlabel('xlabel')
        ax.set_title('title title title')
        ax.set_xlim()
        ax.set_ylim()
        ax.set_xlim([-1, 2])
        ax.set_ylim([-2, 4])
        ax.set_xlim(1, 2)
        ax.set_ylim(1, 4)
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xscale('log')
        ax.set_yscale('log')

    def test_axes_xyaxes_warnings(self):
        ax = Axes()
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)
        warnings_expected = 8
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warnings.
            ax.set_xscale('symlog', keyword='unrecognized')  # 2 warnings
            ax.set_xscale('logit')  # 1 warning
            ax.set_xscale('unrecognized')  # 1 warning
            ax.set_yscale('symlog', keyword='unrecognized')  # 2 warnings
            ax.set_yscale('logit')  # 1 warning
            ax.set_yscale('unrecognized')  # 1 warning
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        self.printv('      test_axes_xyaxes_warnings: tried to make probelmatic calls to Axes lim & scale methods '
                    'and got {}/{} warnings. ax = {}'.format(len(w), warnings_expected, ax))

    def test_axes_aspect(self):
        ax = Axes()
        ax.plot([0, 10, 0, 1])
        ax.set_aspect('equal')
        ax.set_aspect(16.0/9.0)

        # Test warnings
        warnings_expected = 3
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            ax.set_aspect('equal', adjustable='datalim', anchor='C', share=True)
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        self.printv('      test_axes_aspect: tried to call Axes set_aspect() using unimplemented keywords '
                    'and got {}/{} warnings. ax = {}'.format(len(w), warnings_expected, ax))

    def test_axes_clear(self):
        ax = Axes()
        ax.plot(self.y, self.x)  # Switch them so the test doesn't get bored.
        ax.clear()
        # Should add something to try to get the number of objects on the test and assert that there are none

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
