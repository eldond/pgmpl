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
import warnings

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
        Figure(tight_layout=True)
        fig1.close()

    def test_figure_warnings(self):
        warnings_expected = 2
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warnings.
            fig = Figure(frameon=True, linewidth=1, edgecolor='k')  # 1 warning
            fig.suptitle('suptitle', unrecognized='keyword')  # 1 warning
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        assert isinstance(fig, Figure)  # It should still return the instance using the implemented keywords.
        self.printv('      test_figure_warnings: tried to initialize Figure with problematic settings '
                    'and got {}/{} warnings.'.format(len(w), warnings_expected))

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
        self.assertRaises(NotImplementedError, fig.add_subplot, 1, 1, 1, projection='polar')
        self.assertRaises(ValueError, fig.add_subplot, 2, 2, 7)
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
        fig2 = Figure(tight_layout=True)
        fig2.set_subplotpars(sp)
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

    def test_catch_deleted_axes(self):
        """Set up the case where gca() is used when Qt has deleted the axes and test robustness"""
        import sip
        # Add an axes instance to a figure
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        # Confirm that gca() recovers this set of axes with no problem and that the figure knows about them
        ax2 = fig.gca()
        self.assertIs(ax, ax2)
        self.assertIsNotNone(fig.axes)
        # Try to delete the axes in Qt, not going through pgmpl in a way that would tell it of the deletion
        sip.delete(ax)
        # Check that fig's reference to the deleted axes is erased by the _deleted_axes_protection() method
        fig._deleted_axes_protection('testing')
        self.assertIsNone(fig.axes)

        # Try again, less directly
        ax = fig.add_subplot(1, 1, 1)
        ax2 = fig.gca()
        self.assertIs(ax, ax2)
        self.assertIsNotNone(fig.axes)
        sip.delete(ax)
        # Confirm that gca() can't find the old axes anymore, because the deleted axes protection found the problem
        ax3 = fig.gca()
        self.assertIsNot(ax, ax3)

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
