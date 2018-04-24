#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.axes but using PyQtGraph to make the plots
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import copy

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg

# pyqtmpl
from translate import plotkw_translator, color_translator, setup_pen_kw
from util import printd


class Axes(pg.PlotItem):
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, **kwargs):
        super(Axes, self).__init__(**kwargs)
        self.legend = self.legend1

    def plot(self, *args, **kwargs):
        try:
            super(Axes, self).plot(*args, **plotkw_translator(**kwargs))
        except AttributeError:
            # Happens when pg.PlotItem.plot() attempts self.legend.addItem() at the END of plot(),
            # so the rest of plot completes.
            pass

    def set_xlabel(self, label):
        self.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        self.setLabel('left', text=label)

    def axhline(self, value, **kwargs):
        self.addLine(y=value, **plotkw_translator(**kwargs))

    def axvline(self, value, **kwargs):
        self.addLine(x=value, **plotkw_translator(**kwargs))

    def errorbar(
            self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None,
            barsabove=None, lolims=None, uplims=None, xlolims=None, xuplims=None,
            errorevery=1, capthick=None, data=None, **kwargs
    ):
        linestyle = kwargs.get('linestyle', kwargs.get('ls', None))

        if data is not None:
            x = data.get('x', None)
            y = data.get('y', None)
            xerr = data.get('xerr', None)
            yerr = data.get('yerr', None)

        if fmt != '':
            kwargs['fmt'] = fmt

        # Separate keywords into those that affect a line through the data and those that affect the errorbars
        ekwargs = copy.deepcopy(kwargs)
        if ecolor is not None:
            ekwargs['color'] = ecolor
        if elinewidth is not None:
            ekwargs['linewidth'] = elinewidth
        epgkw = plotkw_translator(**ekwargs)
        w = [True if i % int(round(errorevery)) == 0 else False for i in range(len(np.atleast_1d(x)))]

        # Draw the line above the errorbars
        if linestyle not in [' '] and not barsabove:
            self.plot(x, y, **kwargs)

        # Draw the errorbars
        def prep(v):
            v = np.atleast_1d(v)
            xx = np.atleast_1d(x)
            n = len(xx)
            if len(v) == n:
                return v[w]
            elif len(v) == 1:
                return v[0] + xx[w]*0

        errb = pg.ErrorBarItem(
            x=prep(x), y=prep(y),
            height=0 if yerr is None else prep(yerr)*2,
            width=0 if xerr is None else prep(xerr)*2,
            **epgkw
        )
        self.addItem(errb)

        # Draw the caps. pyqtgraph does the caps differently from matplotlib, so we'll put this together manually
        # instead of using pyqtgraph ErrorBarItem's caps.
        if ((capsize is not None) and (capsize <= 0)) or ((capthick <= 0) and (capthick is not None)):
            printd('  Axes.errorbar no caps')
        else:
            capkw = copy.deepcopy(kwargs)
            capkw['linestyle'] = ' '
            if capsize is not None:
                capkw['markersize'] = capsize
            if capthick is not None:
                capkw['markeredgewidth'] = capthick
            if yerr is not None and np.atleast_1d(yerr).max() > 0:
                if uplims and lolims:
                    capkw['marker'] = '^'
                    self.plot(x, y + yerr, **capkw)
                    capkw['marker'] = 'v'
                    self.plot(x, y - yerr, **capkw)
                elif uplims:
                    capkw['marker'] = 'v'
                    self.plot(x, y - yerr, **capkw)
                elif lolims:
                    capkw['marker'] = '^'
                    self.plot(x, y + yerr, **capkw)
                else:  # Neither lolims nor uplims
                    capkw['marker'] = '_'
                    self.plot(x, y + yerr, **capkw)
                    self.plot(x, y - yerr, **capkw)

            if xerr is not None and np.atleast_1d(xerr).max() > 0:
                if xuplims and xlolims:
                    capkw['marker'] = '>'
                    self.plot(x + xerr, y, **capkw)
                    capkw['marker'] = '<'
                    self.plot(x - xerr, y, **capkw)
                elif xuplims:
                    capkw['marker'] = '<'
                    self.plot(x - xerr, y, **capkw)
                elif xuplims:
                    capkw['marker'] = '>'
                    self.plot(x + xerr, y, **capkw)
                else:  # Neither xuplims nor xlolims
                    capkw['marker'] = '|'
                    self.plot(x + xerr, y, **capkw)
                    self.plot(x - xerr, y, **capkw)

        # OR draw the line below the errorbars
        if linestyle not in [' '] and barsabove:
            self.plot(x, y, **kwargs)
        return errb

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, data=None, **kwargs):

        # Set up xy data
        if data is not None:
            x = data['x']
            y1 = data['y1']
            y2 = data['y2']

        x = np.atleast_1d(x)
        y1 = np.atleast_1d(y1)
        y2 = np.atleast_1d(y2)
        if len(y2) == 1:
            y2 += x*0

        # Set up colors and display settings
        ekw = copy.deepcopy(kwargs)
        ekw['color'] = ekw.pop('edgecolor', ekw.pop('color', 'k'))

        if 'facecolor' in kwargs:
            brush = color_translator(color=kwargs['facecolor'], alpha=kwargs.get('alpha', None))
        elif 'color' in kwargs:
            brush = color_translator(color=kwargs['color'], alpha=kwargs.get('alpha', None))
        else:
            brush = color_translator(color='b', alpha=kwargs.get('alpha', None))
        printd('  pyqtmpl.axes.Axes.fill_between(): brush = {}, ekw = {}, setup_pen_kw(**ekw) = {}'.format(
            brush, ekw, setup_pen_kw(**ekw)))

        # Handle special keywords
        if where is not None:
            if interpolate:
                warnings.warn('Warning: interpolate keyword to fill_between is not handled yet.')
            d = np.diff(np.append(0, where))
            start_i = np.where(d == 1)[0]
            end_i = np.where(d == -1)[0]
            if len(end_i) < len(start_i):
                end_i = np.append(end_i, len(d))
            printd('  fill_between where: start_i = {}, end_i = {}'.format(start_i, end_i))

        else:
            start_i = [0]
            end_i = [len(x)]

        if step is not None:
            warnings.warn('Warning: step keyword to fill_between is not handled yet.')

        # Do plot
        fb = []
        for i in range(len(start_i)):
            si = start_i[i]
            ei = end_i[i]
            fb += [pg.FillBetweenItem(
                pg.PlotDataItem(x[si:ei], y1[si:ei]),
                pg.PlotDataItem(x[si:ei], y2[si:ei]),
                pen=setup_pen_kw(**ekw),
                brush=brush,
            )]
            self.addItem(fb[i])

        return fb

    def legend1(self, *args, **kwargs):
        def addItem(*args, **kwargs):
            return None
        printd('  custom legend call')
        p = super(Axes, self)
        leg = p.addLegend()
        children = p.getViewBox().allChildren()

        for child in children:
            if hasattr(child, 'name') and child.name() is not None \
                    and hasattr(child, 'isVisible') and child.isVisible():
                leg.addItem(child, child.name())
        return leg
