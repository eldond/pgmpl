#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.axes but using PyQtGraph to make the plots.

Classes and methods imitate Matplotlib counterparts as closely as possible, so please see Matplotlib documentation for
more information.
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import copy
import unittest

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg
from matplotlib import rcParams
from cycler import cycler
from collections import defaultdict

# pgmpl
from translate import plotkw_translator, color_translator, setup_pen_kw, color_map_translator, symbol_translator
from util import printd, tolist, is_iterable, is_numeric
from text import Text


class Axes(pg.PlotItem):
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, **kwargs):
        sharex = kwargs.pop('sharex', None)
        sharey = kwargs.pop('sharey', None)
        super(Axes, self).__init__(**kwargs)
        self.legend = Legend(ax=self)
        self.prop_cycle = rcParams['axes.prop_cycle']
        tmp = self.prop_cycle()
        self.cyc = defaultdict(lambda: next(tmp))
        self.prop_cycle_index = 0
        if sharex is not None:
            self.setXLink(sharex)
        if sharey is not None:
            self.setYLink(sharey)

    def clear(self):
        printd('  Clearing Axes instance {}...'.format(self))
        super(Axes, self).clear()
        self.legend.clear()
        self.prop_cycle_index = 0

    def plot(self, *args, **kwargs):
        """
        Translates arguments and keywords to matplotlib.axes.Axes.plot() method so they can be passed to
        pg.PlotItem.plot() instead.
        :param args: Plot arguments
        :param kwargs: Plot keywords
        :return: plotItem instance
        """
        need_cycle = any([k not in kwargs.keys() for k in self.prop_cycle.keys])
        if need_cycle:
            printd('keys needed', list(self.prop_cycle.keys), level=2)
            cur = self.cyc
            for k in self.prop_cycle.keys:
                if k not in kwargs.keys():
                    kwargs[str(k)] = cur[self.prop_cycle_index][k]
                    printd('kwargs["{}"] = {}'.format(k, kwargs[str(k)]), level=2)
            self.prop_cycle_index += 1
            if self.prop_cycle_index > len(self.prop_cycle):
                self.prop_cycle_index = 0
        return super(Axes, self).plot(*args, **plotkw_translator(**kwargs))

    def scatter(
            self, x, y, s=10, c=None, marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
            linewidths=None, verts=None, edgecolors=None, data=None, **kwargs
    ):
        """
        Translates arguments and keywords for matplotlib.axes.Axes.scatter() method so they can be passed to pyqtgraph.
        :param x: array like with length n
        :param y: array like with length n
        :param s: scalar or array like with length n, optional
        :param c: color, sequence, or sequence of color, optional
        :param marker: string, optional
        :param cmap: string, optional
        :param norm: Normalize class instance, optional
        :param vmin: scalar, optional
        :param vmax: scalar, optional
        :param alpha: scalar, optional
        :param linewidths: scalar or array like with length n, optional
        :param verts: sequence of (x, y), optional
        :param edgecolors: color or sequence of color
        :param data: dict, optional
        :param kwargs:
        :return: plotItem instance
        """
        if data is not None:
            x = data.get('x')
            y = data.get('y')
            s = data.get('s', None)
            c = data.get('c', None)
            edgecolors = data.get('edgecolors', None)
            linewidths = data.get('linewidths', None)
            # The following keywords are apparently valid within `data`,
            # but they'd conflict with `c`, so they've been neglected:   color facecolor facecolors
        n = len(x)

        # Translate face colors
        if c is None:
            # All same default color
            brush_colors = [color_translator(color='b')] * n
        elif is_numeric(tolist(c)[0]):
            brush_colors = color_map_translator(
                c, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, clip=kwargs.pop('clip', False),
                ncol=kwargs.pop('N', 256), alpha=alpha,
            )
        else:
            # Assume that c is a list/array of colors
            brush_colors = [color_translator(color=cc) for cc in tolist(c)]

        # Translate edge colors
        if edgecolors is None:
            brush_edges = [color_translator(color='k')] * n
        else:
            brush_edges = [color_translator(color=edgecolor, alpha=alpha) for edgecolor in tolist(edgecolors)]

        # Make the lists of symbol settings the same length as x for cases where only one setting value was provided
        if (len(tolist(brush_colors)) == 1) and (n > 1):
            brush_colors = tolist(brush_colors) * n
        if (len(tolist(brush_edges)) == 1) and (n > 1):
            brush_edges = tolist(brush_edges) * n
        if linewidths is not None and (len(tolist(linewidths)) == 1) and (n > 1):
            linewidths = tolist(linewidths) * n

        # Catch & translate other keywords
        kwargs['markersize'] = s
        if marker is not None:
            kwargs['marker'] = marker
        plotkw = plotkw_translator(**kwargs)

        # Fill in keywords we already prepared
        sympen_kw = [{'color': cc} for cc in brush_edges]
        if linewidths is not None:
            for i in range(n):
                sympen_kw[i]['width'] = linewidths[i]
        plotkw['pen'] = None
        plotkw['symbolBrush'] = [pg.mkBrush(color=cc) for cc in brush_colors]
        plotkw['symbolPen'] = [pg.mkPen(**spkw) for spkw in sympen_kw]
        if marker is None:
            verts_x = np.array([vert[0] for vert in verts])
            verts_y = np.array([vert[1] for vert in verts])
            plotkw['symbol'] = pg.arrayToQPath(verts_x, verts_y, connect='all')

        return super(Axes, self).plot(x=x, y=y, **plotkw)

    def set_xlabel(self, label):
        """Imitates basic use of matplotlib.axes.Axes.set_xlabel()"""
        self.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        """Imitates basic use of matplotlib.axes.Axes.set_ylabel()"""
        self.setLabel('left', text=label)

    def set_title(self, label):
        """Imitates basic use of matplotlib.axes.Axes.set_title()"""
        self.setTitle(label)

    def text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
        """
        Imitates matplotlib.axes.Axes.text
        :param x: scalar
        :param y: scalar
        :param s: str
        :param fontdict: dict, optional
        :param withdash: bool, optional
        :param kwargs:
        :return: Text instance
        """
        if withdash:
            warnings.warn('  pgmpl.Axes.text withdash=True keyword is not not handled (yet?)')
        text = Text(x, y, s, fontproperties=fontdict, **kwargs)
        self.addItem(text)
        return text

    def axhline(self, value, **kwargs):
        """Direct imitation of matplotlib axhline"""
        return self.addLine(y=value, **plotkw_translator(**kwargs))

    def axvline(self, value, **kwargs):
        """Direct imitation of matplotlib axvline"""
        return self.addLine(x=value, **plotkw_translator(**kwargs))

    def errorbar(
            self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None,
            barsabove=None, lolims=None, uplims=None, xlolims=None, xuplims=None,
            errorevery=1, capthick=None, data=None, **kwargs
    ):
        """
        Imitates matplotlib.axes.Axes.errorbar
        :return: pyqtgraph.ErrorBarItem instance
            Does not include the line through nominal values as would be included in matplotlib's errorbar; this is
            drawn, but it is a separate object.
        """
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

        # Draw the line below the errorbars
        if linestyle not in [' '] and not barsabove:
            self.plot(x, y, **kwargs)

        # Draw the errorbars
        def prep(v):
            """
            Prepares a value so it has the appropriate dimensions with proper filtering to respect errorevery keyword
            :param v: x, y, xerr, or yerr value or values
            :return: properly dimensioned and filtered array corresponding to v
            """
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

        if kwargs.get('mew', None) is not None:
            capthick = kwargs.pop('mew')
        if kwargs.get('markeredgewidth', None) is not None:
            capthick = kwargs.pop('markeredgewidth')

        # Draw the caps. pyqtgraph does the caps differently from matplotlib, so we'll put this together manually
        # instead of using pyqtgraph ErrorBarItem's caps.
        if ((capsize is not None) and (capsize <= 0)) or ((capthick <= 0) and (capthick is not None)):
            printd('  Axes.errorbar no caps')
        else:
            capkw = copy.deepcopy(kwargs)
            capkw.pop('pg_label', None)
            capkw.pop('label', None)
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

        # OR draw the line above the errorbars
        if linestyle not in [' '] and barsabove:
            self.plot(x, y, **kwargs)

        return errb

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, data=None, **kwargs):
        """
        Imitates matplotlib.axes.Axes.fill_between
        :return: list of pyqtgraph.FillBetweenItem instances
            If the where keyword is not used or has no effect, this will be a list of one item. If where splits the
            range into n segments, then the list will have n elements.
        """
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
        printd('  pgmpl.axes.Axes.fill_between(): brush = {}, ekw = {}, setup_pen_kw(**ekw) = {}'.format(
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

    def set_xlim(self, left=None, right=None, emit=True, auto=False, **kw):
        """Direct imitation of matplotlib set_xlim"""
        if right is None and len(np.atleast_1d(left)) == 2:
            new_xlims = tuple(left)  # X limits were passed in as first argument
        elif right is not None and left is not None \
                and len(np.atleast_1d(left)) == 1 and len(np.atleast_1d(right)) == 1:
            new_xlims = (left, right)
        else:
            new_xlims = None

        if not emit:
            warnings.warn('emit keyword to set_xlim is not handled yet')
        if auto:
            warnings.warn('auto keyword to set_xlim is not handled yet')
        if len(kw.keys()):
            warnings.warn('set_xlim ignores any extra keywords in **kw')

        if new_xlims is not None:
            self.setXRange(new_xlims[0], new_xlims[1])
        return new_xlims

    def set_ylim(self, bottom=None, top=None, emit=True, auto=False, **kw):
        """Direct imitation of matplotlib set_ylim"""
        if top is None and len(np.atleast_1d(bottom)) == 2:
            new_ylims = tuple(bottom)  # Y limits were passed in as first argument
        elif top is not None and bottom is not None \
                and len(np.atleast_1d(bottom)) == 1 and len(np.atleast_1d(top)) == 1:
            new_ylims = (bottom, top)
        else:
            new_ylims = None

        if not emit:
            warnings.warn('emit keyword to set_ylim is not handled yet')
        if auto:
            warnings.warn('auto keyword to set_ylim is not handled yet')
        if len(kw.keys()):
            warnings.warn('set_ylim ignores any extra keywords in **kw')

        if new_ylims is not None:
            self.setYRange(new_ylims[0], new_ylims[1])
        return new_ylims

    def set_xscale(self, value, **kwargs):
        if value == 'linear':
            self.setLogMode(x=False)
        elif value == 'log':
            self.setLogMode(x=True)
        elif value == 'symlog':
            warnings.warn('symlog scaling is not supported')
        elif value == 'logit':
            warnings.warn('logistic transform scaling is not supported')
        else:
            warnings.warn('Unrecognized scale value received by set_xscale: {}. '
                          'Please try again with "linear" or "log".'.format(value))
        if len(kwargs.keys()):
            warnings.warn('Keywords to set_xscale were ignored.')

    def set_yscale(self, value, **kwargs):
        if value == 'linear':
            self.setLogMode(y=False)
        elif value == 'log':
            self.setLogMode(y=True)
        elif value == 'symlog':
            warnings.warn('symlog scaling is not supported')
        elif value == 'logit':
            warnings.warn('logistic transform scaling is not supported')
        else:
            warnings.warn('Unrecognized scale value received by set_yscale: {}. '
                          'Please try again with "linear" or "log".'.format(value))
        if len(kwargs.keys()):
            warnings.warn('Keywords to set_yscale were ignored.')


class Legend:
    """
    Post-generated legend for pgmpl.axes.Axes. This is not a direct imitation of Matplotlib's Legend as it has to
    accept events from pyqtgraph. It also has to bridge the gap between Matplotlib style calling legend after plotting
    and pyqtgraph style calling legend first.

    The call method is supposed to imitate matplotlib.axes.Axes.legend(), though. Bind a Legend class instance to an
    Axes instance as Axes.legend = Legend() and then call Axes.legend() as in matplotlib.
    The binding is done in Axes class __init__.
    """
    def __init__(self, ax=None):
        """
        :param ax: Axes instance
            Reference to the plot axes to which this legend is attached (required).
        """
        # noinspection PyProtectedMember
        from pyqtgraph.graphicsItems.ViewBox.ViewBox import ChildGroup
        self.ax = ax
        # pyqtgraph legends just don't work with some items yet. Avoid errors by trying to use these classes as handles:
        self.unsupported_item_classes = [
            pg.graphicsItems.FillBetweenItem.FillBetweenItem,
            pg.InfiniteLine,
            ChildGroup,
        ]
        #   File "/lib/python2.7/site-packages/pyqtgraph/graphicsItems/LegendItem.py", line 149, in paint
        #     opts = self.item.opts
        # AttributeError: 'InfiniteLine' object has no attribute 'opts'
        self.items_added = []
        self.drag_enabled = True
        self.leg = None

    def supported(self, item):
        """Quick test for whether or not item (which is some kind of plot object) is supported by this legend class"""
        return not any([isinstance(item, uic) for uic in self.unsupported_item_classes])

    @staticmethod
    def handle_info(handles, comment=None):
        """For debugging: prints information on legend handles"""
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
        """
        Adds a legend to the plot axes. This class should be added to axes as they are created so that calling it acts
        like a method of the class and adds a legend, imitating matplotlib legend calling.
        """
        printd('  custom legend call')
        self.leg = self.ax.addLegend()
        # ax.addLegend modifies ax.legend, so we have to put it back in order to
        # preserve a reference to pgmpl.axes.Legend.
        self.ax.legend = self

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
                self.leg.addItem(handle, label)
        return self

    def addItem(self, item, name=None):
        """
        pyqtgraph calls this method of legend and so it must be provided.
        :param item: plot object instance
        :param name: string
        """
        self.items_added += [(item, name)]  # This could be used in place of the search for items, maybe.
        return None

    def draggable(self, on_off=True):
        """
        Provided for compatibility with matplotlib legends, which have this method.
        pyqtgraph legends are always draggable.
        :param on_off: bool
            Throws a warning if user attempts to disable draggability
        """
        self.drag_enabled = on_off
        if not on_off:
            warnings.warn(
                'Draggable switch is not enabled yet. The draggable() method is provided to prevent failures when '
                'plotting routines are converted from matplotlib. pyqtgraph legends are draggable by default.'
            )
        return None

    def clear(self):
        """Removes the legend from Axes instance"""
        printd('  Clearing legend {}...'.format(self.leg))
        try:
            self.leg.scene().removeItem(self.leg)  # https://stackoverflow.com/a/42794442/6605826
        except AttributeError:
            pass


class TestPgmplAxes(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest axes
    """

    verbose = False

    x = np.linspace(0, 1.8, 30)
    y = x**2 + 2.5
    z = x**3 - x**2 * 1.444

    def test_axes_init(self):
        ax = Axes()
        if self.verbose:
            print('test_axes_init: ax = {}'.format(ax))

    def test_axes_plot(self):
        ax = Axes()
        ax.plot(self.x, self.y, color='r')
        if self.verbose:
            print('test_axes_plot: ax = {}'.format(ax))

    def test_axes_scatter(self):
        ax = Axes()
        ax.scatter(self.x, self.y, c=self.z)
        ax.scatter(self.x, self.y, c='b')
        ax.scatter(self.x, self.y, c=self.z, cmap='plasma', marker='s', linewidths=1, edgecolors='r')
        # noinspection PyTypeChecker
        ax.scatter(self.x, self.x*0, c=self.x, cmap='jet', marker=None,
                   verts=[(0, 0), (0.5, 0.5), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, -0.5)])

    def test_axes_err(self):
        ax = Axes()
        yerr = self.y*0.1
        ax.errorbar(self.x, self.y, yerr, color='r')
        ax.fill_between(self.x, -self.y-yerr-1, -self.y+yerr-1)
        if self.verbose:
            print('test_axes_err: ax = {}'.format(ax))

    def test_axes_lines(self):
        ax = Axes()
        ax.axhline(0.5, linestyle='--', color='k')
        ax.axvline(0.5)
        ax.axvline(0.75, linestyle='-', color='b')
        if self.verbose:
            print('test_axes_lines: ax = {}'.format(ax))

    def test_axes_xyaxes(self):
        ax = Axes()
        ax.plot([0, 1], [1, 2])
        ax.set_ylabel('ylabel')
        ax.set_xlabel('xlabel')
        ax.set_title('title title title')
        ax.set_xlim([-1, 2])
        ax.set_ylim([-2, 4])
        ax.set_xscale('linear')
        ax.set_yscale('log')
        if self.verbose:
            print('test_axes_xyaxes: ax = {}'.format(ax))

    def test_axes_clear(self):
        ax = Axes()
        ax.plot(self.y, self.x)  # Switch them so the test doesn't get bored.
        ax.clear()
        # Should add something to try to get the number of objects on the test and assert that there are none
        if self.verbose:
            print('test_axes_clear: ax = {}'.format(ax))

    def test_Legend(self):
        ax = Axes()
        line = ax.plot(self.x, self.y, label='y(x) plot')
        leg = ax.legend()
        leg.addItem(line, name='yx plot')
        leg.draggable()
        leg.clear()
        if self.verbose:
            print('test_axes_Legend: ax = {}, leg = {}'.format(ax, leg))


if __name__ == '__main__':
    unittest.main()
