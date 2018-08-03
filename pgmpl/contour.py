#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.contour but using PyQtGraph to make the plots.

Classes and methods imitate Matplotlib counterparts as closely as possible, so please see Matplotlib documentation for
more information.
"""

# Basic imports
from __future__ import print_function, division
import warnings
import copy

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg
from pyqtgraph import functions as fn

# pgmpl
# noinspection PyUnresolvedReferences
import __init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.translate import plotkw_translator, color_translator, setup_pen_kw, color_map_translator, dealias
from pgmpl.util import printd, tolist, is_numeric


class ContourSet(object):

    def __init__(self, ax, *args, **kwargs):
        self.ax = ax
        pop_default_none = [
            'levels', 'linewidths', 'linestyles', 'alpha', 'origin', 'extent', 'cmap', 'colors', 'norm', 'vmin', 'vmax',
            'antialiased', 'locator',
        ]
        for pdn in pop_default_none:
            self.__setattr__(pdn, kwargs.pop(pdn, None))
        self.filled = kwargs.pop('filled', False)
        self.hatches = kwargs.pop('hatches', [None])
        self.extend = kwargs.pop('extend', 'neither')
        if self.antialiased is None and self.filled:
            self.antialiased = False
        self.nchunk = kwargs.pop('nchunk', 0)
        self.x, self.y, self.z, self.levels = self.choose_xyz_levels(*args)
        self.auto_range(self.z)
        self.draw()

        return

    def auto_range(self, z):
        pad = (z.max() - z.min()) * 0.025
        self.vmin = z.min()+pad if self.vmin is None else self.vmin
        self.vmax = z.max()-pad if self.vmax is None else self.vmax

    def auto_pick_levels(self, z, nlvl=None):
        """
        Pick contour levels automatically
        :param z: 2D array
        :param nlvl: int or None
            Number of levels; set to some arbitrary default if None
        :return: array
        """
        nlvl = 5 if nlvl is None else nlvl
        self.auto_range(z)
        return np.linspace(self.vmin, self.vmax, nlvl)

    def choose_xyz_levels(self, *args):
        """
        Interprets args to pick the contour value Z, the X,Y coordinates, and the contour levels.
        :param args: list of arguments received by __init__
            Could be [z] or [x, y, z] or [z, L] or [x, y, z, L], and L could be an array of levels or a number of levels
        :return: tuple of arrays for x, y, z, and levels
        """
        x = y = lvlinfo = None

        if len(args) == 1:
            z = args[0]
        elif len(args) == 2:
            z, lvlinfo = args
        elif len(args) == 3:
            x, y, z = args
        elif len(args) == 4:
            x, y, z, lvlinfo = args
        else:
            raise TypeError('choose_xyz_levels takes 1, 2, 3, or 4 arguments. Got {} arguments.'.format(len(args)))

        levels = lvlinfo if ((lvlinfo is not None) and np.iterable(lvlinfo)) else self.auto_pick_levels(z, lvlinfo)

        if x is None:
            x, y = np.arange(np.shape(z)[0]), np.arange(np.shape(z)[1])

        return x, y, z, levels

    def extl(self, v):
        """
        Casts input argument as a list and ensures it is at least as long as levels
        :param v: Some variable
        :return: List of values for variable v; at least as long as self.levels
        """
        return tolist(v) * int(np.ceil(len(self.levels) / len(tolist(v))))

    def draw(self):
        if self.colors is None:
            # Assign color map
            self.colors = color_map_translator(
                self.levels, **{a: self.__getattribute__(a) for a in ['alpha', 'cmap', 'norm', 'vmin', 'vmax']})
        else:
            self.colors = self.extl(self.colors)

        if self.filled:
            raise NotImplementedError('Filled contours with contourf not yet finished.')
        else:
            self.draw_unfilled()

    def draw_unfilled(self):
        lws, lss = self.extl(self.linewidths), self.extl(self.linestyles)
        pens = [setup_pen_kw(penkw=dict(color=self.colors[i]), linestyle=lss[i], linewidth=lws[i])
                for i in range(len(self.levels))]
        contours = [pg.IsocurveItem(data=self.z, level=lvl, pen=pens[i]) for i, lvl in enumerate(self.levels)]
        x0, y0, x1, y1 = self.x.min(), self.y.min(), self.x.max(), self.y.max()
        for contour in contours:
            contour.translate(x0, y0)  # https://stackoverflow.com/a/51109935/6605826
            contour.scale((x1 - x0) / np.shape(self.z)[0], (y1 - y0) / np.shape(self.z)[1])
            self.ax.addItem(contour)


class QuadContourSet(ContourSet):
    """Provided to make this thing follow the same sort of structure as matplotlib"""
    def __init__(self, *args, **kwargs):
        super(QuadContourSet, self).__init__(*args, **kwargs)
