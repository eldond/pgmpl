#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.axes but using PyQtGraph to make the plots
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg

# pyqtmpl
from translate import plotkw_translator


class Axes(pg.PlotItem):
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, **kwargs):
        super(Axes, self).__init__(**kwargs)

    def plot(self, *args, **kwargs):
        super(Axes, self).plot(*args, **plotkw_translator(**kwargs))

    def set_xlabel(self, label):
        self.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        self.setLabel('left', text=label)

    def axhline(self, value, **kwargs):
        self.addLine(y=value, **plotkw_translator(**kwargs))

    def axvline(self, value, **kwargs):
        self.addLine(x=value, **plotkw_translator(**kwargs))
