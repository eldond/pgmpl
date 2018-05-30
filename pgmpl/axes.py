#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.axes but using PyQtGraph to make the plots.

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
from text import Text


class Axes(pg.PlotItem):
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, **kwargs):
        self.nrows = kwargs.pop('nrows', 1)
        self.ncols = kwargs.pop('ncols', 1)
        self.index = kwargs.pop('index', 1)
        super(Axes, self).__init__(**kwargs)
        self.legend = Legend(ax=self)
        self.prop_cycle = rcParams['axes.prop_cycle']
        tmp = self.prop_cycle()
        self.cyc = defaultdict(lambda: next(tmp))
        self.prop_cycle_index = 0
        if kwargs.get('sharex', None) is not None:
            self.setXLink(kwargs['sharex'])
        if kwargs.get('sharey', None) is not None:
            self.setYLink(kwargs['sharey'])

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

    @staticmethod
    def _prep_scatter_colors(n, c=None, **kwargs):
        """Helper function to prepare colors for scatter plot"""
        edgecolors = kwargs.pop('edgecolors', None)
        # Translate face colors
        if c is None:
            # All same default color
            brush_colors = [color_translator(color='b')] * n
        elif is_numeric(tolist(c)[0]):
            brush_colors = color_map_translator(c, **kwargs)
        else:
            # Assume that c is a list/array of colors
            brush_colors = [color_translator(color=cc) for cc in tolist(c)]

        # Translate edge colors
        if edgecolors is None:
            brush_edges = [color_translator(color='k')] * n
        else:
            brush_edges = [color_translator(color=edgecolor, alpha=kwargs.get('alpha', None))
                           for edgecolor in tolist(edgecolors)]

        # Make the lists of symbol settings the same length as x for cases where only one setting value was provided
        if (len(tolist(brush_colors)) == 1) and (n > 1):
            brush_colors = tolist(brush_colors) * n
        if (len(tolist(brush_edges)) == 1) and (n > 1):
            brush_edges = tolist(brush_edges) * n

        return brush_colors, brush_edges

    def scatter(
            self, x=None, y=None, s=10, c=None, marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
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

        brush_colors, brush_edges = self._prep_scatter_colors(
            n, c=c, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, edgecolors=edgecolors, alpha=alpha, **kwargs)

        # Make the lists of symbol settings the same length as x for cases where only one setting value was provided
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

    def imshow(self, x, aspect=None, **kwargs):
        if aspect is not None:
            self.set_aspect(aspect, adjustable='box')
        img = AxesImage(x, **kwargs)
        self.addItem(img)
        return img

    def set_xlabel(self, label):
        """Imitates basic use of matplotlib.axes.Axes.set_xlabel()"""
        self.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        """Imitates basic use of matplotlib.axes.Axes.set_ylabel()"""
        self.setLabel('left', text=label)

    def set_title(self, label):
        """Imitates basic use of matplotlib.axes.Axes.set_title()"""
        self.setTitle(label)

    def set_aspect(self, aspect, adjustable=None, **kw):
        vb = self.getViewBox()
        if aspect == 'equal':
            vb.setAspectLocked(lock=True, ratio=1)
        elif aspect == 'auto':
            vb.setAspectLocked(lock=False)
        else:
            vb.setAspectLocked(lock=True, ratio=aspect)
        if adjustable not in ['box', None]:
            warnings.warn('Axes.set_aspect ignored keyword: adjustable')
        if kw.pop('anchor', None) is not None:
            warnings.warn('Axes.set_aspect ignored keyword: anchor')
        if kw.pop('share', False):
            warnings.warn('Axes.set_aspect ignored keyword: share')

    def text(self, x, y, s, **kwargs):
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
        if kwargs.pop('withdash', False):
            warnings.warn('  pgmpl.Axes.text withdash=True keyword is not not handled (yet?)')
        text = Text(x, y, s, fontproperties=kwargs.pop('fontdict', None), **kwargs)
        self.addItem(text)
        return text

    def axhline(self, value, **kwargs):
        """Direct imitation of matplotlib axhline"""
        return self.addLine(y=value, **plotkw_translator(**kwargs))

    def axvline(self, value, **kwargs):
        """Direct imitation of matplotlib axvline"""
        return self.addLine(x=value, **plotkw_translator(**kwargs))

    def _draw_errbar_caps(self, x, y, **capkw):
        """
        Helper function for errorbar.
        Draw caps on errorbars. pyqtgraph does the caps differently from matplotlib, so we'll put this together
        manually instead of using pyqtgraph ErrorBarItem's caps.
        :param x, y, xerr, yerr: Input data
        :param capsize: Size of error bar caps (sets size of markers)
        :param capkw: deepcopy of kwargs passed to errorbar. These are passed to plot when drawing the caps.
        """
        # Remove unused keywords so they don't make trouble later
        capkw.pop('pg_label', None)
        capkw.pop('label', None)
        # Extract keywords
        capsize = capkw.pop('capsize', None)
        capthick = capkw.pop('capthick', None)
        xerr = capkw.pop('xerr', None)
        yerr = capkw.pop('yerr', None)

        # Setup
        capkw['linestyle'] = ' '
        if capsize is not None:
            capkw['markersize'] = capsize
        if capthick is not None:
            capkw['markeredgewidth'] = capthick

        if yerr is not None and np.atleast_1d(yerr).max() > 0:
            capkw['marker'] = 'v' if capkw.pop('lolims', None) else '_'
            self.plot(x, y - yerr, **capkw)
            capkw['marker'] = '^' if capkw.pop('uplims', None) else '_'
            self.plot(x, y + yerr, **capkw)

        if xerr is not None and np.atleast_1d(xerr).max() > 0:
            capkw['marker'] = '<' if capkw.pop('xlolims', None) else '|'
            self.plot(x - xerr, y, **capkw)
            capkw['marker'] = '>' if capkw.pop('xuplims', None) else '|'
            self.plot(x + xerr, y, **capkw)

    @staticmethod
    def _sanitize_errbar_data(x, y=None, xerr=None, yerr=None, mask=None):
        """
        Helper function for errorbar.
        Forces all data to be the same size and applies filters
        :param x: Independent variable
        :param y, xerr, yerr: Dependent variable and error bars (optional)
        :param mask: bool array selecting which data to keep
        :return: stuple of sanitized x, y, xerr, yerr

        """

        def prep(v):
            """
            Prepares a value so it has the appropriate dimensions with proper filtering to respect errorevery keyword
            :param v: x, y, xerr, or yerr value or values
            :return: properly dimensioned and filtered array corresponding to v
            """
            if v is None:
                return None
            v = np.atleast_1d(v)
            xx = np.atleast_1d(x)
            n = len(xx)
            if len(v) == n:
                return v[mask]
            elif len(v) == 1:
                return v[0] + xx[mask]*0

        return prep(x), prep(y), prep(xerr), prep(yerr)

    def errorbar(self, x=None, y=None, yerr=None, xerr=None, **kwargs):
        """
        Imitates matplotlib.axes.Axes.errorbar
        :return: pyqtgraph.ErrorBarItem instance
            Does not include the line through nominal values as would be included in matplotlib's errorbar; this is
            drawn, but it is a separate object.
        """
        kwargs = dealias(**kwargs)
        data = kwargs.pop('data', None)

        if data is not None:
            x = data.get('x', None)
            y = data.get('y', None)
            xerr = data.get('xerr', None)
            yerr = data.get('yerr', None)

        # Separate keywords into those that affect a line through the data and those that affect the errorbars
        ekwargs = copy.deepcopy(kwargs)
        if kwargs.get('ecolor', None) is not None:
            ekwargs['color'] = kwargs.pop('ecolor')
        if kwargs.get('elinewidth', None) is not None:
            ekwargs['linewidth'] = wargs.pop('elinewidth')
        epgkw = plotkw_translator(**ekwargs)
        w = np.array([True if i % int(round(kwargs.pop('errorevery', 1))) == 0 else False
                      for i in range(len(np.atleast_1d(x)))])

        # Draw the line below the errorbars
        if kwargs.get('linestyle', None) not in [' '] and not kwargs.get('barsabove', None):
            self.plot(x, y, **kwargs)

        # Draw the errorbars
        xp, yp, xerrp, yerrp = self._sanitize_errbar_data(x, y, xerr, yerr, w)

        errb = pg.ErrorBarItem(
            x=xp, y=yp, height=0 if yerr is None else yerrp*2, width=0 if xerr is None else xerrp*2, **epgkw
        )
        self.addItem(errb)

        if kwargs.get('markeredgewidth', None) is not None:
            kwargs['capthick'] = kwargs.pop('markeredgewidth')

        if ((kwargs.get('capsize', None) is not None) and (kwargs.get('capsize', 0) <= 0)) or \
                ((kwargs.get('capthick', None) is not None) and (kwargs.get('capthick', 0) <= 0)):
            printd('  Axes.errorbar no caps')
        else:
            self._draw_errbar_caps(xp, yp, xerr=xerrp, yerr=yerrp, **copy.deepcopy(kwargs))

        # OR draw the line above the errorbars
        if kwargs.pop('linestyle', None) not in [' '] and kwargs.pop('barsabove', None):
            self.plot(x, y, **kwargs)

        return errb

    def fill_between(self, x=None, y1=None, y2=0, **kwargs):
        """
        Imitates matplotlib.axes.Axes.fill_between
        :return: list of pyqtgraph.FillBetweenItem instances
            If the where keyword is not used or has no effect, this will be a list of one item. If where splits the
            range into n segments, then the list will have n elements.
        """
        # Set up xy data
        data = kwargs.pop('data', None)
        if data is not None:
            x = data['x']
            y1 = data['y1']
            y2 = data['y2']

        x = np.atleast_1d(x)
        y1 = np.atleast_1d(y1)
        y2 = np.atleast_1d(y2)
        if len(y2) == 1:
            y2 = x*0 + y2

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
        if kwargs.get('where', None) is not None:
            if kwargs.pop('interpolate', False):
                warnings.warn('Warning: interpolate keyword to fill_between is not handled yet.')
            d = np.diff(np.append(0, kwargs['where']))
            start_i = np.where(d == 1)[0]
            end_i = np.where(d == -1)[0]
            if len(end_i) < len(start_i):
                end_i = np.append(end_i, len(d))
            printd('  fill_between where: start_i = {}, end_i = {}'.format(start_i, end_i))

        else:
            start_i = [0]
            end_i = [len(x)]

        if kwargs.pop('step', None) is not None:
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

    def set_xlim(self, left=None, right=None, **kw):
        """Direct imitation of matplotlib set_xlim"""
        if right is None and len(np.atleast_1d(left)) == 2:
            new_xlims = tuple(left)  # X limits were passed in as first argument
        elif right is not None and left is not None \
                and len(np.atleast_1d(left)) == 1 and len(np.atleast_1d(right)) == 1:
            new_xlims = (left, right)
        else:
            new_xlims = None

        if not kw.pop('emit', True):
            warnings.warn('emit keyword to set_xlim is not handled yet')
        if kw.pop('auto', False):
            warnings.warn('auto keyword to set_xlim is not handled yet')
        if len(kw.keys()):
            warnings.warn('set_xlim ignores any extra keywords in **kw')

        if new_xlims is not None:
            self.setXRange(new_xlims[0], new_xlims[1])
        return new_xlims

    def set_ylim(self, bottom=None, top=None, **kw):
        """Direct imitation of matplotlib set_ylim"""
        if top is None and len(np.atleast_1d(bottom)) == 2:
            new_ylims = tuple(bottom)  # Y limits were passed in as first argument
        elif top is not None and bottom is not None \
                and len(np.atleast_1d(bottom)) == 1 and len(np.atleast_1d(top)) == 1:
            new_ylims = (bottom, top)
        else:
            new_ylims = None

        if not kw.pop('emit', True):
            warnings.warn('emit keyword to set_ylim is not handled yet')
        if kw.pop('auto', False):
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


class AxesImage(pg.ImageItem):
    def __init__(self, x, **kwargs):
        data = kwargs.pop('data', None)
        cmap = kwargs.pop('cmap', None)
        norm = kwargs.pop('norm', None)
        alpha = kwargs.pop('alpha', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        extent = kwargs.pop('extent', None)
        origin = kwargs.pop('origin', None)

        if data is not None:
            x = data['x']
            if len(data.keys()) > 1:
                warnings.warn('Axes.imshow does not extract keywords from data yet (just x).')

        xs = copy.copy(x)

        self.check_inputs(**kwargs)

        if origin in ['upper', None]:
            xs = xs[::-1]
            if extent is None:
                extent = (-0.5, x.shape[1]-0.5, -(x.shape[0]-0.5), -(0-0.5))
        else:
            if extent is None:
                extent = (-0.5, x.shape[1]-0.5, -0.5, x.shape[0]-0.5)

        if len(np.shape(xs)) == 3:
            xs = np.transpose(xs, (2, 0, 1))
        else:
            xs = np.array(color_map_translator(
                xs.flatten(), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, clip=kwargs.pop('clip', False),
                ncol=kwargs.pop('N', 256), alpha=alpha,
            )).T.reshape([4] + tolist(xs.shape))

        super(AxesImage, self).__init__(np.transpose(xs))
        if extent is not None:
            self.resetTransform()
            self.translate(extent[0], extent[2])
            self.scale((extent[1] - extent[0]) / self.width(), (extent[3] - extent[2]) / self.height())

        self.cmap = cmap
        self.norm = norm
        self.alpha = alpha
        self.vmin = x.min() if vmin is None else vmin
        self.vmax = x.max() if vmax is None else vmax

    @staticmethod
    def check_inputs(**kw):
        if kw.pop('shape', None) is not None:
            warnings.warn('Axes.imshow ignored keyword: shape. I could not get this working with matplotlib, '
                          'so I had nothing to emulate.')
        if kw.pop('imlim', None) is not None:
            warnings.warn('Axes.imshow ignored keyword: imlim.')
        if kw.pop('interpolation', None) is not None:
            warnings.warn('Axes.imshow ignored keyword: interpolation.')
        if kw.pop('filternorm', 1) != 1 or kw.pop('filterrad', 4.0) != 4.0:
            warnings.warn('Axes.imshow ignores changes to keywords filternorm and filterrad.')
        if kw.pop('resample', None) is not None:
            warnings.warn('Axes.imshow ignored keyword: resample.')
        if kw.pop('url', None) is not None:
            warnings.warn('Axes.imshow ignored keyword: url.')
        if len(kw.keys()):
            warnings.warn('Axes.imshow got unhandled keywords: {}'.format(kw.keys()))


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

    def __call__(self, handles=None, labels=None, **kw):
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

        unhandled_kws = dict(loc=None,
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
        )
        for unhandled in unhandled_kws.keys():
            if unhandled in kw.keys():
                kw.pop(unhandled)
                warnings.warn(
                    'pgmpl.axes.Legend.__call__ got unhandled keyword: {}. This keyword might be implemented later.')
        if len(kw.keys()):
            warnings.warn('pgmpl.axes.Legend.__call__ got unrecognized keywords: {}'.format(kw.keys()))

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
            printd('  Could not clear legend (maybe it is already invisible?')
