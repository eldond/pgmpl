#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import os
import numpy as np
import unittest


def printd(*args, **kw):
    """
    Prints only if debug flag is turned on (greater than level)
    :param args: Things to print
    :param level: int
        Debugging level
    """
    debug = os.environ.get('PGMPL_DEBUG', "0")
    if int(debug) >= kw.pop('level', 1):
        print(*args)


def tolist(x):
    return np.ndarray.tolist(np.atleast_1d(x))


class TestPgmplUtil(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest util
    """

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


if __name__ == '__main__':
    unittest.main()
