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

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

if os.environ.get('PYQTMPL_DEBUG', None) is None:
    os.environ['PYQTMPL_DEBUG'] = "0"
app = QtGui.QApplication(sys.argv)
