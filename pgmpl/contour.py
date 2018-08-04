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
            self.draw_filled()
        else:
            self.draw_unfilled()

    @staticmethod
    def _isocurve2plotcurve(curve):
        """
        Converts an IsocuveItem instance to a PlotCurveItem instance so it can be used with FillBetweenItem

        FAILS because the curves aren't sorted properly to allow good connections between segments. That is, a contour
        can break where it intersects the edge of the plot/data range, and restart later where it re-enters. These entry
        and exit points are reconnected arbitrarily or incorrectly.

        :param curve: IsocurveItem instance
        :return: PlotCurveItem with the same path
        """
        curve.generatePath()
        new_curve = pg.PlotCurveItem()
        new_curve.path = curve.path
        return new_curve

    def _scale_contour_lines(self, lines):
        """
        Translates and stretches contour lines
        :param lines: A list of path segments. e.g. [[(x, y), (x, y)], [(x, y), (x, y)]]
        :return: A list of path segments, shifted and stretched to fit the x, y data range.
        """
        x0, y0, x1, y1 = self.x.min(), self.y.min(), self.x.max(), self.y.max()
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                newx = lines[i][j][0] * (x1 - x0) / np.shape(self.z)[0] + x0
                newy = lines[i][j][1] * (y1 - y0) / np.shape(self.z)[1] + y0
                lines[i][j] = (newx, newy)
        return lines

    def _detect_direction(self, x, y):
        """
        Tries to decide whether path goes CCW or not

        :param x: array of numbers
        :param y: array of numbers
        :return: True if path appears to run CCW
        """

        if len(x) < 2:
            return True  # No need to reverse if 0 or 1 element

        x0, y0 = np.mean([self.x.min(), self.x.max()]), np.mean([self.y.min(), self.y.max()])
        theta = np.arctan2(y-y0, x-x0)

        # Find theta increment
        dth = np.diff(theta)

        # Unwrap theta increment so there are no jumps as theta loops around
        mdth = np.median(dth)
        jumps = (np.sign(dth) == -np.sign(mdth)) & (abs(dth) > (2*abs(mdth)))
        for jump in np.where(jumps)[0]:
            dth[jump] -= 2*np.pi * np.sign(dth[jump])

        return np.mean(dth) > 0

    def _join_lines(self, lines):
        """
        Joins segments of a broken path. Use for contours which cross out of bounds and back in. Has to find the right
        direction of path segments and join them in the right order.

        :param lines: List of path segments. e.g. [[(x, y), (x,y)], [(x, y), (x, y)]]
        :return: Flattened list of correctly joined path segments e.g. [(x, y), (x,y), (x, y), (x, y)], running CCW
        """
        for i in range(len(lines)):
            # Make sure all segments run CCW within themselves
            x, y = map(list, zip(*lines[i]))
            if not self._detect_direction(x, y):
                lines[i] = lines[i][::-1]
        # Make sure the set of start points of each segment runs CCW
        x1 = np.array([line[0][0] for line in lines])
        y1 = np.array([line[0][1] for line in lines])
        if not self._detect_direction(x1, y1):
            lines = lines[::-1]
        return [point for line in lines for point in line]

    def _close_curve(self, line):
        """
        Closes a curve which may be open between the two end points because it intersects the edge of the plot
        :param line: List of vertices along a CCW path. e.g. [(x, y), (x, y), ...] such as output by _join_lines()
        :return: List of vertices along a closed CCW path
        """

        if all(np.atleast_1d(line[0] == line[-1])):
            # Path is already closed; return it with no changes
            return line

        # Determine which endpoints are on which edges
        edge0 = np.append(line[0][0] == np.array([self.x.min(), self.x.max()]),
                          line[0][1] == np.array([self.y.min(), self.y.max()]))
        edge1 = np.append(line[-1][0] == np.array([self.x.min(), self.x.max()]),
                          line[-1][1] == np.array([self.y.min(), self.y.max()]))
        same_edge = edge0 == edge1

        if (not any(edge0) and not any(edge1)) or all(same_edge):
            # Endpoints are not on the edges, or are on the same edge. Just close the loop.
            newline = line + [line[0]]
        else:
            # Endpoints are not on the same edge; complicated closure

            # Get the boundary path
            boundary = [
                (self.x.min(), self.y.min()),
                (self.x.max(), self.y.min()),
                (self.x.max(), self.y.max()),
                (self.x.min(), self.y.max()),
            ]
            # Find which boundary path points are between the endpoints of the curve
            x0, y0 = np.mean([self.x.min(), self.x.max()]), np.mean([self.y.min(), self.y.max()])
            theta0 = np.arctan2(line[0][1] - y0, line[0][0] - x0)
            theta1 = np.arctan2(line[-1][1] - y0, line[-1][0] - x0)
            thetab = np.array([np.arctan2(b[1] - y0, b[0] - x0) for b in boundary])
            # Make sure the thing wraps the right way; we are continuing from point -1 back to point 0
            theta0 = theta0 + 2*np.pi if theta0 < theta1 else theta0
            thetab = np.array([b + (2*np.pi if b < theta1 else 0) for b in thetab])
            # Add in corners of the data range boundary to complete the curve
            newline = line + tolist(np.array(boundary)[thetab < theta0])
            # And finally, close it
            newline += [line[0]]
        return newline

    def draw_filled(self):
        pens = [setup_pen_kw(penkw=dict(color=self.colors[i])) for i in range(len(self.levels))]
        for i in range(len(self.levels)):
            lines = fn.isocurve(self.z, self.levels[i], connected=True, extendToEdge=True)[::-1]
            lines = self._scale_contour_lines(lines)
            oneline = self._join_lines(lines)
            oneline = self._close_curve(oneline)

            x, y = map(list, zip(*oneline))
            curve = pg.PlotDataItem(x, y, pen=pens[i])
            self.ax.addItem(curve)
            if i > 0:
                fill = pg.FillBetweenItem(curve, prev_curve, brush=pg.mkBrush(color=self.colors[i]))
                self.ax.addItem(fill)  # doesn't work well
            prev_curve = curve

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
