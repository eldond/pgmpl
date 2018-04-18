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

# pyqtmpl
from translate import plotkw_translator


class Axes:
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, fig=None, *args, **kwargs):
        if fig is None:
            fig = figure()
        self.fig = fig
        self.axes = self.fig.layout.addPlot(*args, **kwargs)

    def plot(self, *args, **kwargs):
        self.axes.plot(*args, **plotkw_translator(**kwargs))

    def set_xlabel(self, label):
        self.axes.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        self.axes.setLabel('left', text=label)

    def axhline(self, value, **kwargs):
        self.axes.addLine(y=value, **plotkw_translator(**kwargs))

    def axvline(self, value, **kwargs):
        self.axes.addLine(x=value, **plotkw_translator(**kwargs))
