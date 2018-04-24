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
from util import printd, tolist


class Axes(pg.PlotItem):
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, **kwargs):
        super(Axes, self).__init__(**kwargs)
        self.legend = Legend(ax=super(Axes, self))

    def plot(self, *args, **kwargs):
        return super(Axes, self).plot(*args, **plotkw_translator(**kwargs))

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


class Legend:
    def __init__(self, ax=None):
        from pyqtgraph.graphicsItems.ViewBox.ViewBox import ChildGroup
        self.ax = ax
        # pyqtgraph legends just don't work with some items yet. Avoid errors by trying to use these classes as handles:
        self.unsupported_item_classes = [
            pg.graphicsItems.FillBetweenItem.FillBetweenItem,
            ChildGroup,
        ]
        self.items_added = []

    def supported(self, item):
        return not any([isinstance(item, uic) for uic in self.unsupported_item_classes])

    @staticmethod
    def handle_info(handles, comment=None):
        if comment is not None:
            printd(comment)
        for i, handle in enumerate(tolist(handles)):
            printd('  {i:02d} handle name: {name:}, class: {cls:}, isVisible: {isvis:}'.format(
                i=i,
                name=handle.name() if hasattr(handle, 'name') else None,
                cls=handle.__class__ if hasattr(handle, '__class__') else ' not found ',
                isvis=handle.isVisible() if hasattr(handle, 'isVisible') else None,
            ))

    def __call__(
            self,
            handles=None,
            labels=None,
            loc=None,
            numpoints=None,    # the number of points in the legend line
            markerscale=None,  # the relative size of legend markers vs. original
            markerfirst=True,  # controls ordering (left-to-right) of legend marker and label
            scatterpoints=None,    # number of scatter points
            scatteryoffsets=None,
            prop=None,          # properties for the legend texts
            fontsize=None,        # keyword to set font size directly
            # spacing & pad defined as a fraction of the font-size
            borderpad=None,      # the whitespace inside the legend border
            labelspacing=None,   # the vertical space between the legend entries
            handlelength=None,   # the length of the legend handles
            handleheight=None,   # the height of the legend handles
            handletextpad=None,  # the pad between the legend handle and text
            borderaxespad=None,  # the pad between the axes and legend border
            columnspacing=None,  # spacing between columns
            ncol=1,     # number of columns
            mode=None,  # mode for horizontal distribution of columns. None, "expand"
            fancybox=None,  # True use a fancy box, false use a rounded box, none use rc
            shadow=None,
            title=None,  # set a title for the legend
            framealpha=None,  # set frame alpha
            edgecolor=None,  # frame patch edgecolor
            facecolor=None,  # frame patch facecolor
            bbox_to_anchor=None,  # bbox that the legend will be anchored.
            bbox_transform=None,  # transform for the bbox
            frameon=None,  # draw frame
            handler_map=None,
    ):
        printd('  custom legend call')
        leg = self.ax.addLegend()

        if handles is None:
            handles = self.ax.getViewBox().allChildren()
            self.handle_info(handles, comment='handles from allChildren')
            handles = [item for item in handles if hasattr(item, 'isVisible') and item.isVisible()]
        else:
            handles = tolist(handles)

        nlab = len(np.atleast_1d(labels))
        if labels is not None and nlab == 1:
            labels = tolist(labels)*len(handles)
        elif labels is not None and nlab == len(handles):
            labels = tolist(labels)
        else:
            handles = [item for item in handles if hasattr(item, 'name') and item.name() is not None]
            labels = [item.name() for item in handles]

        for handle, label in zip(handles, labels):
            if self.supported(handle):
                leg.addItem(handle, label)
        return leg

    def addItem(self, item, name=None):
        self.items_added += [(item, name)]
        return None
