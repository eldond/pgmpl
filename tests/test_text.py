#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for figure.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.text import Text

class TestPgmplText(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest text
    """

    verbose = False

    def test_text_simple(self):
        t = Text(0.5, 0.5, 'text1')
        assert isinstance(t, Text)

    def test_text_on_plot(self):
        from pgmpl.pyplot import subplots
        fig, ax = subplots(1)
        ax.plot([0, 1], [1, 0])
        ax.plot([0, 1], [0, 1])
        ax.text(0.5, 0.9, 'text1')
        ax.text(0.5, 0.7, 'text2', color='r', va='bottom', ha='left')
        ax.text(0.5, 0.5, 'text3', color='b', va='top', ha='right')
        ax.text(0.5, 0.3, 'text4', color='g', va='center', ha='center')


if __name__ == '__main__':
    unittest.main()
