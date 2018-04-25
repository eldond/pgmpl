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
from matplotlib import rcParams

# pyqtmpl imports
from info import *  # Defines __version__, etc.
from util import printd
from translate import color_translator

__all__ = ['figure', 'axes', 'pyplot', 'translate', 'examples']

# Handle debugging
if os.environ.get('PYQTMPL_DEBUG', None) is None:
    os.environ['PYQTMPL_DEBUG'] = "0"


def set_debug(enable=True):
    os.environ['PYQTMPL_DEBUG'] = str(int(enable))


# Setup style, etc.
pg.setConfigOption('background', color_translator(**{'color': rcParams['axes.facecolor']}))
pg.setConfigOption('foreground', color_translator(**{'color': rcParams['axes.edgecolor']}))

# Make a QApp so that windows can be opened
app = QtGui.QApplication.instance()
if app is None:
    printd('No pre-existing QApplication found. Creating one...')
    app = QtGui.QApplication(sys.argv)
else:
    printd('Using pre-existing QApplication.')
