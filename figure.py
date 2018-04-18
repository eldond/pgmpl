#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.figure but using PyQtGraph to make the plots
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
from axes import Axes


class Figure:
    """
    Imitates matplotlib.figure.Figure using PyQtGraph
    """
    def __init__(
            self, figsize=None, dpi=None,
            facecolor=None, edgecolor=None, linewidth=0.0,
            frameon=True, subplotpars=None, tight_layout=None, constrained_layout=None,
    ):
        self.patch_resize = True  # Controls whether resize events mess with margins or not
        pg.setConfigOption('background', 'w' if facecolor is None else facecolor)
        pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsWindow()
        self.win.resizeEvent_original = self.win.resizeEvent
        self.win.resizeEvent = self.resize_event
        if figsize is None:
            figsize = (800, 600)
        self.width = figsize[0]
        self.height = figsize[1]
        self.win.resize(self.width, self.height)
        self.axes = None
        self.layout = pg.GraphicsLayout()
        self.win.setCentralItem(self.layout)
        self.tight = tight_layout or constrained_layout
        if self.tight:
            self.margins = {'left': 10, 'top': 10, 'right': 10, 'bottom': 10, 'hspace': 10, 'wspace': 10}
        else:
            self.margins = None

        if subplotpars is not None:
            self.set_subplotpars(subplotpars)

        if frameon is not False and linewidth > 0 and edgecolor:
            warnings.warn('WARNING: frame around figure edge is not implemented yet')
        if dpi is not None:
            warnings.warn('WARNING: keyword DPI to class Figure is ignored.')

    def resize_event(self, event):
        if hasattr(event, 'size'):
            self.width = event.size().width()
            self.height = event.size().height()
            if self.patch_resize and not self.tight:
                self.set_subplotpars(None)
        self.win.resizeEvent_original(event)

    def set_subplotpars(self, pars):
        fx = self.width
        fy = self.height
        if pars is not None:
            self.margins = {
                'left': pars.left, 'top': pars.top, 'right': pars.right, 'bottom': pars.bottom,
                'hspace': pars.hspace, 'wspace': pars.wspace,
            }
        if self.margins is not None:
            if self.tight:
                self.layout.setContentsMargins(
                    self.margins['left'], self.margins['top'], self.margins['right'], self.margins['bottom'],
                )
                self.layout.setSpacing((self.margins['wspace'] + self.margins['hspace'])/2.0)
            else:
                nrows = 3  # This isn't actually known, so we have to just guess
                ncols = 3
                spx = (self.margins['right'] - self.margins['left'])/nrows * self.margins['wspace'] * fx
                spy = (self.margins['top'] - self.margins['bottom'])/ncols * self.margins['hspace'] * fy
                self.layout.setSpacing((spx + spy)/2.0)
                self.layout.setContentsMargins(
                    self.margins['left']*fx,
                    (1-self.margins['top'])*fy,
                    (1-self.margins['right'])*fx,
                    self.margins['bottom']*fy,
                )

    def add_subplot(self, nrows, ncols, index, projection=None, polar=None, **kwargs):
        if projection is not None and projection != 'rectilinear':
            raise NotImplementedError('projection keyword in add_subplot is not ready')
        if polar is not None and polar is not False:
            raise NotImplementedError('polar projection is not ready')
        row = int(np.floor(index/ncols))
        if row > (nrows-1):
            raise ValueError('index {} would be on row {}, but the last row is {}!'.format(index, row, nrows-1))
        col = index % ncols
        ax = Axes(self, row, col, **kwargs)
        return ax
