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
from pgmpl.util import tolist, printd, is_iterable, set_debug


class TestPgmplUtil(unittest.TestCase):
    """
    Each test function tests one of the functions in translate.py.
    The test function names are test_ + the name of the function they test.
    """

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_set_debug(self):
        set_debug()
        assert os.environ.get('PGMPL_DEBUG', None) == "1"
        set_debug(False)
        assert os.environ.get('PGMPL_DEBUG', None) == "0"
        set_debug(True)
        assert os.environ.get('PGMPL_DEBUG', None) == "1"
        set_debug(0)
        assert os.environ.get('PGMPL_DEBUG', None) == "0"
        set_debug(1)
        assert os.environ.get('PGMPL_DEBUG', None) == "1"
        set_debug(None)
        assert os.environ.get('PGMPL_DEBUG', None) == "0"

    def test_printd(self):
        test_string_1 = '\nthis string should print, but the other string should not'
        test_string_2 = '\nthis string should NOT print, but the other string SHOULD'
        debug = os.environ.get('PGMPL_DEBUG', "0")
        os.environ['PGMPL_DEBUG'] = "1"
        printd(test_string_1)
        printd('this-should-print:', 'test-item-1a', 'test_item_2a_in-list-of-things', 5, 6, 'more-things-in-the-list')
        os.environ['PGMPL_DEBUG'] = "0"
        printd(test_string_2)
        printd('SHOULD-NOT-PRINT:', 'test-item-1b', 'testitem2b-in-a-listofthings', 5, 6.1, 'morelistlol', 'blah')
        os.environ['PGMPL_DEBUG'] = debug  # Put it back how it was (polite~~)

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

    def test_is_iterable(self):
        assert is_iterable([1, 2, 3])
        assert is_iterable(np.array([1, 2, 3]))
        assert is_iterable({'ralph': 'blah', 'bob': 'spork'})
        assert is_iterable('blah')
        assert not is_iterable(1)
        assert not is_iterable(0.1)
        assert not is_iterable(None)

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
