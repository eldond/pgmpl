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

        # Get the boundary path
        self.boundary = [
            (self.x.min(), self.y.min()),
            (self.x.max(), self.y.min()),
            (self.x.max(), self.y.max()),
            (self.x.min(), self.y.max()),
        ]
        self.draw()

        return

    def auto_range(self, z):
        pad = (z.max() - z.min()) * 0.025
        self.vmin = z.min()+pad if self.vmin is None else self.vmin
        self.vmax = z.max() + (pad if self.filled else -pad) if self.vmax is None else self.vmax

    def auto_pick_levels(self, z, nlvl=None):
        """
        Pick contour levels automatically
        :param z: 2D array
        :param nlvl: int or None
            Number of levels; set to some arbitrary default if None
        :return: array
        """
        nlvl = 8 if nlvl is None else nlvl
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
            x, y = np.arange(np.shape(z)[1]), np.arange(np.shape(z)[0])

        return x, y, z.T, levels

    def _extl(self, v):
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
            self.colors = self._extl(self.colors)

        if self.filled:
            self.draw_filled()
        else:
            self.draw_unfilled()

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

    def _find_which_edge(self, xy_point):
        print(xy_point, 'xy_point')
        a = np.where([xy_point[0] == self.x.max(), xy_point[1] == self.y.max(),
                      xy_point[0] == self.x.min(), xy_point[1] == self.y.min()])[0]
        if len(a):
            return a[0]
        else:
            return 4

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

        if not lines:
            print('return early because not lines:', lines)
            return tolist(np.array(self.boundary)[np.array([0, 1, 2, 3, 0])])

        corners = np.roll(self.boundary, -2)

        def get_more_corners(next_e0_, edge_):
            print('edge: {}, next_edge start: {}'.format(edge_, next_e0_))
            corners_ = []
            if (next_e0_ - edge_) >= 1:
                print('new corner = ', corners[edge_])
                corners_ += [tuple(corners[edge_]), tuple(corners[edge_+1])]
                print('Add corner {}'.format(corners[edge_], corners[edge_+1]))
            if (next_e0_ - edge_) >= 2:
                corners_ += [tuple(corners[edge_+2])]
                print('Add corner {}'.format(corners[edge_+2]))
            if (next_e0_ - edge_) >= 3:
                print('Add corner {}'.format(corners[edge_+3]))
                corners_ += [tuple(corners[edge_+3])]#, tuple(corners[edge_])]
            return corners_

        oneline = lines.pop(0)
        #if not lines:
        #    return oneline

        while len(lines):
            # Find which edge each segment starts/ends on
            edge = self._find_which_edge(oneline[-1])
            edge0 = np.array([self._find_which_edge(line[0]) for line in lines])
            if not any(edge0 >= edge):
                edge -= 4
            eligible = [line for line, e0 in zip(lines, edge0) if e0 >= edge]
            next_line = eligible[edge0[edge0 >= edge].argmin()]
            next_e0 = edge0[edge0 >= edge].min()
            oneline += get_more_corners(next_e0, edge)

            lines.remove(next_line)
            oneline += next_line

        e0 = self._find_which_edge(oneline[0])
        e1 = self._find_which_edge(oneline[-1])
        if e1 > e0:
            e1 -= 4
        print('e0, e1', e0, e1)
        more_corners = get_more_corners(e0, e1)
        print('more corners = ', more_corners)
        oneline += get_more_corners(e0, e1)
        oneline += [oneline[0]]
        return oneline

    def _close_curve(self, line):  # Remove during cleanup -----------------------------------
        """
        OLD IDEA; DON'T USE ANYMORE
        Closes a curve which may be open between the two end points because it intersects the edge of the plot
        :param line: List of vertices along a CCW path. e.g. [(x, y), (x, y), ...] such as output by _join_lines()
        :return: List of vertices along a closed CCW path
        """

        if not line:
            # Empty line; nothing to do
            return self.boundary

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

            # Find which boundary path points are between the endpoints of the curve
            x0, y0 = np.mean([self.x.min(), self.x.max()]), np.mean([self.y.min(), self.y.max()])
            theta0 = np.arctan2(line[0][1] - y0, line[0][0] - x0)
            theta1 = np.arctan2(line[-1][1] - y0, line[-1][0] - x0)
            thetab = np.array([np.arctan2(b[1] - y0, b[0] - x0) for b in self.boundary])
            # Make sure the thing wraps the right way; we are continuing from point -1 back to point 0
            theta0 = theta0 + 2*np.pi if theta0 < theta1 else theta0
            thetab = np.array([b + (2*np.pi if b < theta1 else 0) for b in thetab])
            # Add in corners of the data range boundary to complete the curve
            newline = line + tolist(np.array(self.boundary)[thetab < theta0])
            # And finally, close it
            newline += [line[0]]
        return newline

    def draw_filled(self):
        pens = [setup_pen_kw(penkw=dict(color=self.colors[i])) for i in range(len(self.levels))]
        use_pen = pg.mkPen(color='k')
        curves = [None] * (len(self.levels))
        joined_lines = [None] * (len(self.levels))
        for i in range(len(self.levels)):
            print('level # {}, @ {}'.format(i, self.levels[i]))
            lines = fn.isocurve(self.z, self.levels[i], connected=True, extendToEdge=True)[::-1]
            lines = self._scale_contour_lines(lines)
            oneline = self._join_lines(lines)
            print('oneline ', oneline)

            joined_lines[i] = oneline
            x, y = map(list, zip(*oneline))
            curves[i] = pg.PlotDataItem(x, y, pen=use_pen)#pens[i])
            #if i == len(self.levels)-1:
            #    self.ax.addItem(curve, pen=pg.mkPen(color='r'))
            #if i == len(self.levels)-2:

            # self.ax.addItem(curve)
            # if i == 0:
            #     x0 = np.mean([point[0] for point in oneline])
            #     y0 = np.mean([point[1] for point in oneline])
            #     xc = [x0, x0 + 1e-12]
            #     yc = [y0, y0 + 1e-12]
            #     curve_c = pg.PlotDataItem(xc, yc, pen=pens[i])
            #     fill = pg.FillBetweenItem(curve, curve_c, brush=pg.mkBrush(color=self.colors[i]))
            #     self.ax.addItem(fill)
            # else:  # i > 0:
            # #elif i == len(self.levels)-1:
            #     fill = pg.FillBetweenItem(curve, prev_curve, brush=pg.mkBrush(color=self.colors[i]))
            #     self.ax.addItem(fill)
            # prev_curve = curve

        # Get the curves at the edges of the array
        x0 = np.mean([point[0] for point in joined_lines[1]])
        y0 = np.mean([point[1] for point in joined_lines[1]])
        dx0 = np.std([point[0] for point in joined_lines[1]])
        dy0 = np.std([point[1] for point in joined_lines[1]])
        x1 = np.mean([point[0] for point in joined_lines[-2]])
        y1 = np.mean([point[1] for point in joined_lines[-2]])
        dx1 = np.std([point[0] for point in joined_lines[-2]])
        dy1 = np.std([point[1] for point in joined_lines[-2]])
        if (dx1**2 + dy1**2) > (dx0**2 + dy0**2):
            curves = [pg.PlotDataItem([x0, x0 + 1e-12], [y0, y0 + 1e-12], pen=pens[0])] + curves
            #curves[-1] = pg.PlotDataItem(
            #    np.array([self.x.min(), self.x.max()])[np.array([0, 0, 1, 1, 0])],
            #    np.array([self.y.min(), self.y.max()])[np.array([0, 1, 1, 0, 0])], pen=pens[-1])
        else:
            curves = curves + [pg.PlotDataItem([x1, x1 + 1e-12], [y1, y1 + 1e-12], pen=pens[-1])]
            #curves[0] = pg.PlotDataItem(
            #    np.array([self.x.min(), self.x.max()])[0, 0, 1, 1, 0],
            #    np.array([self.y.min(), self.y.max()])[0, 1, 1, 0, 0], pen=pens[0])

        for j in range(len(self.levels)):
            i = len(self.levels)-j-1
            fill = pg.FillBetweenItem(curves[i], curves[i+1], brush=pg.mkBrush(color=self.colors[i]))
            self.ax.addItem(fill)

    def draw_unfilled(self):
        lws, lss = self._extl(self.linewidths), self._extl(self.linestyles)
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
