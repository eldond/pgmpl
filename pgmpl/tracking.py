#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
For keeping track of things at the module level
"""

# Basic imports
from __future__ import print_function, division
import warnings

# pgmpl
from util import printd


class WTracker:

    def __init__(self):
        self.open_windows = []

    def window_closed(self, win):
        if win in self.open_windows:
            printd('  tracker detected window closed: {}'.format(win))
            self.open_windows.remove(win)
        else:
            warnings.warn('  tracker received notification of closing of untracked window!')
        self.status()

    def window_opened(self, win):
        printd('  tracker detected new window opened: {}'.format(win))
        self.open_windows += [win]
        self.status()

    def status(self):
        printd('  {} tracked windows = {}'.format(len(self.open_windows), self.open_windows))


tracker = WTracker()
