#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.colorbar but using PyQtGraph

Classes and methods imitate Matplotlib counterparts as closely as possible, so please see Matplotlib documentation for
more information.
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg

# pgmpl
import pgmpl.__init__
from pgmpl.util import printd, tolist


class ColorbarBase(object):

    def __init__(
            self, ax, cmap=None, norm=None, alpha=None, values=None, boundaries=None, orientation='vertical',
            ticklocation='auto', extend='neither', spacing='uniform', ticks=None, format=None, drawedges=False,
            filled=True, extendfrac=None, extendrect=False, label='',
    ):
        printd('ColorbarBase.__init__:mappable.vmin={},mappable.vmax={}'.format(self.mappable.vmin, self.mappable.vmax))
        a = np.linspace(0, 1, 256).reshape(256, 1)
        printd('  pgmpl colorbar initializing in {} orientation'.format(orientation))
        if orientation == 'horizontal':
            ylim = [0, 1]
            xlim = [self.mappable.vmin, self.mappable.vmax]
            a = a.T
            show_ax = 'bottom'
        else:
            xlim = [0, 1]
            ylim = [self.mappable.vmin, self.mappable.vmax]
            show_ax = 'right'
        extent = tuple(xlim + ylim)
        ax.imshow(a, cmap=cmap, norm=norm, alpha=alpha, origin='lower', extent=extent)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        for ax_side in ['top', 'bottom', 'right', 'left']:
            if ax_side == show_ax:
                ax.showAxis(ax_side)
            else:
                ax.hideAxis(ax_side)
        ax.setLabel(show_ax, text=label)
        ax.setMouseEnabled(x=False, y=False)


class Colorbar(ColorbarBase):

    def __init__(self, ax, mappable, **kw):
        printd('pgmpl.colorbar.Colorbar.__init__()...')
        self.mappable = mappable
        kw['cmap'] = cmap = mappable.cmap
        kw['norm'] = norm = mappable.norm
        super(Colorbar, self).__init__(ax, **kw)
