#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for colorbar.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np
import warnings

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.colorbar import Colorbar


class TestPgmplColorbar(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    # Make some test data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 5)
    a = x[:, np.newaxis] * y[np.newaxis, :]

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_colorbar(self):
        """Tests that fig.colorbar executes without exceptions and returns the expected class"""
        from pgmpl.pyplot import subplots
        fig, ax = subplots(1)
        img = ax.imshow(self.a)
        cb = fig.colorbar(img)
        assert isinstance(cb, Colorbar)
        fig.close()

    def test_colorbar_warnings(self):
        """Tests that appropriate warnings are issued by fig.colorbar()"""
        from pgmpl.pyplot import subplots
        fig, ax = subplots(1)
        img = ax.imshow(self.a)
        # Check that warnings are issued
        with warnings.catch_warnings(record=True) as w:
            fig.colorbar(img, fake_keyword_for_testing_unimplented_warning='blah')
        assert any(['fake_keyword_for_testing_unimplented_warning' in str(ww) for ww in w])
        fig.close()

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
