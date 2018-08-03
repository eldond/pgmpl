#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Test script for tracking.py
"""

# Basic imports
from __future__ import print_function, division
import os
import unittest
import numpy as np
import copy
import warnings

# pgmpl
from pgmpl import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.tracking import WTracker, tracker
from pgmpl.figure import Figure


class TestPgmplTracking(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    def printv(self, *args):
        if self.verbose:
            print(*args)

    def test_tracker(self):
        assert isinstance(tracker, WTracker)
        tracker.close_all()
        dummy = 'dummy_window'
        open_windows0 = copy.deepcopy(tracker.open_windows)
        tracker.window_opened(dummy)
        assert dummy in tracker.open_windows
        tracker.status()
        open_windows1 = copy.deepcopy(tracker.open_windows)
        assert len(open_windows1) == len(open_windows0) + 1
        tracker.window_closed(dummy)
        tracker.status()
        open_windows2 = copy.deepcopy(tracker.open_windows)
        assert dummy not in tracker.open_windows
        assert len(open_windows1) == len(open_windows2) + 1

    def test_tracker_warnings(self):
        warnings_expected = 1
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            tracker.window_closed('this is obviously not a tracked window, so there should be a warning')
            # Verify that warnings were made.
            assert len(w) == warnings_expected
        self.printv('      test_tracker_warnings: called window_closed with a nonsense window to trigger a warning  '
                    'and got {}/{} warnings.'.format(len(w), warnings_expected))

    def test_close_all(self):
        Figure()
        tracker.close_all()
        assert len(tracker.open_windows) == 0

    def setUp(self):
        test_id = self.id()
        test_name = '.'.join(test_id.split('.')[-2:])
        self.printv('{}...'.format(test_name))

    def tearDown(self):
        test_name = '.'.join(self.id().split('.')[-2:])
        self.printv('    {} done.'.format(test_name))


if __name__ == '__main__':
    unittest.main()
