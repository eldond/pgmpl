#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import sys
import os

# GUI imports
from PyQt4 import QtGui

# Plotting imports
import pyqtgraph as pg
from info import *  # Defines __version__, etc.

__all__ = ['figure', 'axes', 'pyplot', 'translate', 'examples']

# Setup style, etc.
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Make a QApp so that windows can be opened
if os.environ.get('PYQTMPL_DEBUG', None) is None:
    os.environ['PYQTMPL_DEBUG'] = "0"
app = QtGui.QApplication(sys.argv)
