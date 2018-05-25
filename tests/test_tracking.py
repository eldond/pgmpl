#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for figure.py
"""

# Basic imports
from __future__ import print_function, division
import unittest
import numpy as np
import copy

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.tracking import WTracker, tracker


class TestPgmplTracking(unittest.TestCase):

    verbose = False

    def test_tracker(self):
        assert isinstance(tracker, WTracker)
        if self.verbose:
            print('test_tracker: tracker = {}'.format(tracker))
        dummy = 'dummy_window'
        tracker.window_opened(dummy)
        assert dummy in tracker.open_windows
        tracker.status()
        open_windows1 = copy.deepcopy(tracker.open_windows)
        tracker.window_closed(dummy)
        tracker.status()
        open_windows2 = copy.deepcopy(tracker.open_windows)
        assert dummy not in tracker.open_windows
        assert len(open_windows1) == len(open_windows2) + 1


if __name__ == '__main__':
    unittest.main()
