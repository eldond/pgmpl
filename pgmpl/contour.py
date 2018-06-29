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
from matplotlib import rcParams
from collections import defaultdict

# pgmpl
# noinspection PyUnresolvedReferences
import __init__  # __init__ does setup stuff like making sure a QApp exists
from translate import plotkw_translator, color_translator, setup_pen_kw, color_map_translator, dealias
from util import printd, tolist, is_numeric


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
        self.vmin = z.min() if self.vmin is None else self.vmin
        self.vmax = z.max() if self.vmax is None else self.vmax

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

    def draw(self):
        if self.filled:
            self.draw_filled()
        else:
            self.draw_unfilled()

    def draw_filled(self):
        printd(' not ready yet lol')

    def draw_unfilled(self):
        contours = [pg.IsocurveItem(data=self.z, level=level, pen='r') for level in self.levels]
        x0, y0, x1, y1 = self.x.min(), self.y.min(), self.x.max(), self.y.max()
        for contour in contours:
            contour.translate(x0, y0) # https://stackoverflow.com/a/51109935/6605826
            contour.scale((x1 - x0) / np.shape(self.z)[0], (y1 - y0) / np.shape(self.z)[1])
            self.ax.addItem(contour)


class QuadContourSet(ContourSet):
    """Provided to make this thing follow the same sort of structure as matplotlib"""
    def __init__(self, *args, **kwargs):
        super(QuadContourSet, self).__init__(*args, **kwargs)
