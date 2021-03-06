#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for text.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np
import warnings

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.text import Text


class TestPgmplText(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_text_simple(self):
        """Very basic test of just initializing the Text class"""
        t = Text(0.5, 0.5, 'text1')
        assert isinstance(t, Text)
        assert t is t()

    def test_text_on_plot(self):
        """Test setting up a plot and putting text on it with Axes.text, which of course uses the Text class."""
        from pgmpl.pyplot import subplots
        fig, ax = subplots(1)
        ax.plot([0, 1], [1, 0])
        ax.plot([0, 1], [0, 1])
        t1 = ax.text(0.5, 0.9, 'text1')
        t2 = ax.text(0.5, 0.7, 'text2', color='r', va='bottom', ha='left')
        t3 = ax.text(0.5, 0.5, 'text3', color='b', va='top', ha='right')
        t4 = ax.text(0.5, 0.3, 'text4', color='g', va='center', ha='center')
        assert isinstance(t1, Text)
        assert isinstance(t2, Text)
        assert isinstance(t3, Text)
        assert isinstance(t4, Text)
        self.printv('      test_text_on_plot: t1, t2 ,t3, t4 = {}, {}, {}, {}'.format(t1, t2, t3, t4))
        fig.close()

    def test_text_warnings(self):
        """Use all the unsupported keywords to throw warnings"""
        warnings_expected = 7
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            t = Text(0.51, 0.5, 'text', wrap=True, fontproperties={}, fake_keyword=True, linespacing=2, usetex=True,
                     rotation_mode='default', multialignment='left')
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        assert isinstance(t, Text)  # It should still return the instance using the implemented keywords.
        self.printv('      test_text_warnings: tried to make a Text instance using unimplemented keywords '
                    'and got {}/{} warnings. t = {}'.format(len(w), warnings_expected, t))

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
