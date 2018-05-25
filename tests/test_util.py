#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for util.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np
import os

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.util import tolist, printd, is_iterable


class TestPgmplUtil(unittest.TestCase):

    verbose = False

    def test_printd(self):
        test_string_1 = 'this string should print, but the other string should not'
        test_string_2 = 'this string should NOT print, but the other string SHOULD'
        debug = os.environ.get('PGMPL_DEBUG', "0")
        os.environ['PGMPL_DEBUG'] = "1"
        printd(test_string_1)
        os.environ['PGMPL_DEBUG'] = "0"
        printd(test_string_2)
        os.environ['PGMPL_DEBUG'] = debug  # Put it back how it was (polite~~)
        if self.verbose:
            print('  test_printd completed')

    def test_tolist(self):
        ar = np.array([1, 2, 3])
        a2 = np.zeros((3, 3))
        bo = True
        li = [1, 2, 3]
        no = None
        sc = 0
        st = 'blah'
        tu = (1, 2)
        for thing in [ar, a2, bo, li, no, sc, st, tu]:
            assert isinstance(tolist(thing), list)
            assert isinstance(tolist(thing) + [1, 2, 3], list)
        if self.verbose:
            print('  test_tolist completed')

    def test_is_iterable(self):
        assert is_iterable([1, 2, 3])
        assert is_iterable(np.array([1, 2, 3]))
        assert is_iterable({'ralph': 'blah', 'bob': 'spork'})
        assert is_iterable('blah')
        assert not is_iterable(1)
        assert not is_iterable(0.1)
        if self.verbose:
            print('  test_is_iterable completed')


if __name__ == '__main__':
    unittest.main()
