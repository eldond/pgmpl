# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np
import warnings

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.axes import Axes
from pgmpl.legend import Legend


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

    def test_Legend(self):
        """Tests both the legend method of Axes and the Legend class implicitly"""
        ax = Axes()
        line = ax.plot(self.x, self.y, label='y(x) plot')
        leg = ax.legend()
        leg.addItem(line, name='yx plot')
        leg.draggable()
        leg.clear()
        ax2 = Axes()
        ax2.plot(self.x, self.y, color='r', label='y(x) plot red')
        ax2.plot(self.x, -self.y, color='b', label='y(x) plot blue')
        ax2.legend(labels='blah')

        self.printv('test_axes_Legend: ax = {}, leg = {}'.format(ax, leg))

    def test_Legend_warnings(self):
        ax = Axes()
        ax.plot(self.x, self.y, label='y(x) plot')
        leg = ax.legend()

        # Test warnings
        warnings_expected = 5
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger warnings.
            leg.draggable(False)  # 1 warning
            # Trigger more warnings:
            ax.legend(blah='unrecognized keyword should make warning', borderaxespad=5)  # 2 warnings
            ax.legend(loc=0)  # 1 warning
            ax.legend(loc=4)  # 1 warning
            # Verify that warnings were made.
        self.printv('      test_axes_Legend: triggered a warning from Legend and got {}/{} warnings. leg = {}'.format(
            len(w), warnings_expected, leg))
        assert len(w) == warnings_expected

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
