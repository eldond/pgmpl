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
import pgmpl.__init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.translate import plotkw_translator, color_translator, setup_pen_kw, color_map_translator, dealias
from pgmpl.legend import Legend
from pgmpl.util import printd, tolist, is_numeric
from pgmpl.text import Text
from pgmpl.contour import QuadContourSet


class Axes(pg.PlotItem):
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, **kwargs):
        for item in ['sharex', 'sharey']:
            setattr(self, item, kwargs.pop(item, None))
        for item in ['nrows', 'ncols', 'index']:
            setattr(self, item, kwargs.pop(item, 1))
        super(Axes, self).__init__(**kwargs)
        self.legend = Legend(ax=self)
        self.prop_cycle = rcParams['axes.prop_cycle']
        tmp = self.prop_cycle()
        self.cyc = defaultdict(lambda: next(tmp))
        self.prop_cycle_index = 0
        if self.sharex is not None:
            self.setXLink(self.sharex)
        if self.sharey is not None:
            self.setYLink(self.sharey)

    def clear(self):
        """
        Clears the axes

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.clear.html
        """
        printd('  Clearing Axes instance {}...'.format(self))
        super(Axes, self).clear()
        self.legend.clear()
        self.prop_cycle_index = 0

    def plot(self, *args, **kwargs):
        """
        Translates arguments and keywords to matplotlib.axes.Axes.plot() method so they can be passed to
        pg.PlotItem.plot() instead.

        See https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.plot.html

        :param args: Plot arguments
            They will be passed straight through to plot()

        :param kwargs: Plot keywords
            They will be translated from mpl to pg conventions and then passed to plot()

        :return: plotItem instance returned by pg.PlotItem.plot()
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
    def _prep_scatter_colors(n, **kwargs):
        """
        Helper function to prepare colors for scatter plot

        :param n: int
            Number of colors to prepare

        :param kwargs: keywords for controlling color, like
            c
            edgecolors
            alpha

        :return: tuple
            list of colors for primary usage (maybe for face, line, etc.)
            list of colors for edges
        """
        edgecolors = kwargs.pop('edgecolors', None)
        c = kwargs.pop('c', None)
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
        brush_edges = [color_translator(color='k')] * n if edgecolors is None \
            else [color_translator(color=ec, alpha=kwargs.get('alpha', None)) for ec in tolist(edgecolors)]

        # Make the lists of symbol settings the same length as x for cases where only one setting value was provided
        brush_colors = tolist(brush_colors) * n if (len(tolist(brush_colors)) == 1) and (n > 1) else brush_colors
        brush_edges = tolist(brush_edges) * n if (len(tolist(brush_edges)) == 1) and (n > 1) else brush_edges

        return brush_colors, brush_edges

    @staticmethod
    def _make_custom_verts(verts):
        """
        Makes a custom symbol from the verts keyword accepted by scatter

        :param verts: sequence of (x, y)

        :return: str
            The name of the new custom key. It has been "installed" into pyqtgraph's Symbols dictionary.
        """
        from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
        verts_x = np.array([vert[0] for vert in verts])
        verts_y = np.array([vert[1] for vert in verts])
        key = 'custom_pgmpl_symbol_1'
        Symbols[key] = pg.arrayToQPath(verts_x, verts_y, connect='all')
        return key

    def scatter(self, x=None, y=None, **kwargs):
        """
        Translates arguments and keywords for matplotlib.axes.Axes.scatter() method so they can be passed to pyqtgraph.

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.scatter.html

        :param x: array-like
            Values for the X-axis, with length n

        :param y: array-like
            Values for the Y-axis, with length n

        :param s: array-like or scalar [optional]
            Size or sizes for the markers. If iterable, must have length n.

        :param c: color, sequence, or sequence of color [optional]
            Colors for the markers. If specifying more than one color, there must be n colors.

        :param marker: string [optional]
            Code for marker symbol, like 'o'

        :param cmap: string [optional]
            Matplotlib colormap name

        :param norm: Normalize class instance [optional]
            An instance of one of Matplotlib's normalization classes, like mpl.colors.Normalize
            If not provided, a new instance of mpl.colors.Normalize will be created.

        :param vmin: numeric scalar [optional]
            Minimum value for color axis. Used to initialize mpl.colors.Normalize, if needed. Otherwise, ignored.

        :param vmax: numeric scalar [optional]
            Maximum value for color axis. Used to initialize mpl.colors.Normalize, if needed. Otherwise, ignored.

        :param alpha: numeric scalar [optional]
            Opacity, between 0 (invisible) and 1 (fully opaque)

        :param linewidths: numeric scalar or array-like [optional]
            If array-like, must match length n

        :param verts: sequence of (x, y) [optional]
            Used for creating custom symbols. Set marker=None to use this.

        :param edgecolors: color specification or sequence of color specifications
            Colors for edges

        :param data: dict, optional
            Alternative way to specify some keywords. Must contain x and y.
            Can also contain s, c, edgecolors, and linewidths, which will override any duplicates passed in directly.

        :return: plotItem instance created by plot()
        """
        data = kwargs.pop('data', None)
        linewidths = kwargs.pop('linewidths', None)
        if data is not None:
            x = data.get('x')
            y = data.get('y')
            kwargs['s'] = data.get('s', None)
            kwargs['c'] = data.get('c', None)
            kwargs['edgecolors'] = data.get('edgecolors', None)
            linewidths = data.get('linewidths', None)
            # The following keywords are apparently valid within `data`,
            # but they'd conflict with `c`, so they've been neglected:   color facecolor facecolors
        n = len(x)

        brush_colors, brush_edges = self._prep_scatter_colors(n, **kwargs)

        for popit in ['cmap', 'norm', 'vmin', 'vmax', 'alpha', 'edgecolors', 'c']:
            kwargs.pop(popit, None)  # Make sure all the color keywords are gone now that they've been used.

        # Make the lists of symbol settings the same length as x for cases where only one setting value was provided
        if linewidths is not None and (len(tolist(linewidths)) == 1) and (n > 1):
            linewidths = tolist(linewidths) * n

        # Catch & translate other keywords
        kwargs['markersize'] = kwargs.pop('s', 10)
        kwargs.setdefault('marker', 'o')
        plotkw = plotkw_translator(**kwargs)

        # Fill in keywords we already prepared
        sympen_kw = [{'color': cc} for cc in brush_edges]
        if linewidths is not None:
            for i in range(n):
                sympen_kw[i]['width'] = linewidths[i]
        plotkw['pen'] = None
        plotkw['symbolBrush'] = [pg.mkBrush(color=cc) for cc in brush_colors]
        plotkw['symbolPen'] = [pg.mkPen(**spkw) for spkw in sympen_kw]
        plotkw['symbol'] = plotkw.get('symbol', None) or self._make_custom_verts(kwargs.pop('verts', None))
        return super(Axes, self).plot(x=x, y=y, **plotkw)

    def imshow(self, x=None, **kwargs):
        """
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.imshow.html

        :param x: array

        :param aspect: number or str passed to self.set_aspect()

        :param kwargs: other keywords passed to AxesImage()

        :return: AxesImage instance
        """
        aspect = kwargs.pop('aspect', None)
        if aspect is not None:
            self.set_aspect(aspect, adjustable='box')
        img = AxesImage(x, **kwargs)
        self.addItem(img)
        return img

    def contour(self, *args, **kwargs):
        """
        Initializes a QuadContourSet instance for the current Axes with filled=False enforced

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.contour.html

        :return: QuadContourSet instance
        """
        printd('  pgmpl.axes.Axes.contour()...')
        kwargs['filled'] = False
        contours = QuadContourSet(self, *args, **kwargs)
        return contours

    def contourf(self, *args, **kwargs):
        """
        Initializes a QuadContourSet instance for the current Axes with filled=True enforced

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.contourf.html

        :return: QuadContourSet instance
        """
        printd('  pgmpl.axes.Axes.contourf()...')
        kwargs['filled'] = True
        return QuadContourSet(self, *args, **kwargs)

    def set_xlabel(self, label):
        """
        Imitates basic use of matplotlib.axes.Axes.set_xlabel()

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html

        :param label: str
        """
        self.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        """
        Imitates basic use of matplotlib.axes.Axes.set_ylabel()

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html

        :param label: str
        """
        self.setLabel('left', text=label)

    def set_title(self, label):
        """
        Imitates basic use of matplotlib.axes.Axes.set_title()

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_title.html

        :param label: str
        """
        self.setTitle(label)

    def set_aspect(self, aspect, adjustable=None, **kw):
        """
        Sets plot axes aspect ratio

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_aspect.html

        :param aspect: float or str
            'equal': same as 1
            'auto': unlocks aspect ratio so it can change as needed
            number: sets aspect ratio to this value

        :param adjustable: nothing but box has been implemented so far; sorry

        :param kw: Accepts additional keywords to prevent interfacing problems, but can't use them yet.
        """
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

        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.text.html

        :param x: scalar
            X coordinate of displayed text

        :param y: scalar
            Y coordinate of displayed text

        :param s: str
            Text to show

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
        """
        Direct imitation of matplotlib axhline
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axhline.html
        """
        return self.addLine(y=value, **plotkw_translator(**kwargs))

    def axvline(self, value, **kwargs):
        """
        Direct imitation of matplotlib axvline
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axvline.html
        """
        return self.addLine(x=value, **plotkw_translator(**kwargs))

    def _errbar_xcap_mark(self, x, y, errx, **capkw):
        """Draws marks (|, <, or >) at the ends of x error bars"""
        if errx is not None and np.atleast_1d(errx).max() > 0:
            capkw['marker'] = '|<'[int(bool(capkw.pop('xlolims', None)))]
            self.plot(x - errx, y, **capkw)
            capkw['marker'] = '|>'[int(bool(capkw.pop('xuplims', None)))]
            self.plot(x + errx, y, **capkw)

    def _errbar_ycap_mark(self, x, y, erry, **capkw):
        """Draws marks (|, <, or >) at the ends of x error bars"""
        if erry is not None and np.atleast_1d(erry).max() > 0:
            capkw['marker'] = '_v'[int(bool(capkw.pop('ylolims', None)))]
            self.plot(x, y - erry, **capkw)
            capkw['marker'] = '_^'[int(bool(capkw.pop('yuplims', None)))]
            self.plot(x, y + erry, **capkw)

    def _draw_errbar_caps(self, x, y, **capkw):
        """
        Helper function for errorbar.
        Draw caps on errorbars. pyqtgraph does the caps differently from matplotlib, so we'll put this together
        manually instead of using pyqtgraph ErrorBarItem's caps.
        :param x, y, xerr, yerr: Input data
        :param capsize: Size of error bar caps (sets size of markers)
        :param capkw: deepcopy of kwargs passed to errorbar. These are passed to plot when drawing the caps.
        """
        if ((capkw.get('capsize', None) is not None) and (capkw.get('capsize', 0) <= 0)) or \
                ((capkw.get('capthick', None) is not None) and (capkw.get('capthick', 0) <= 0)):
            return
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

        self._errbar_xcap_mark(x, y, xerr, **capkw)
        self._errbar_ycap_mark(x, y, yerr, **capkw)

    @staticmethod
    def _sanitize_errbar_data(x, y=None, xerr=None, yerr=None, mask=None):
        """
        Helper function for errorbar. Does not map to a matplotlib method.

        Forces all data to be the same size and applies filters

        :param x: array
            Independent variable

        :param y, xerr, yerr: arrays
            Dependent variable and error bars (optional)

        :param mask: bool array
            Selector for which data to keep

        :return: tuple of sanitized x, y, xerr, yerr
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

        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.errorbar.html

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
            ekwargs['linewidth'] = kwargs.pop('elinewidth')
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

        self._draw_errbar_caps(xp, yp, xerr=xerrp, yerr=yerrp, **copy.deepcopy(kwargs))

        # OR draw the line above the errorbars
        if kwargs.pop('linestyle', None) not in [' '] and kwargs.pop('barsabove', None):
            self.plot(x, y, **kwargs)

        return errb

    @staticmethod
    def _setup_fill_between_colors(**kwargs):
        """
        Prepares edge plotting keywords and brush for fill_between
        :param kwargs: dictionary of keywords from fill_between
        :return: dict, brush
        """
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
        return ekw, brush

    @staticmethod
    def _setup_fill_between_where(x, **kwargs):
        """
        Handles where and interpolate keywords
        :param x: x values
        :param kwargs: dictionary of keywords received by fill_between
        :return: tuple with two lists of ints giving start and end indices for each segment of data passing where
        """
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
        return start_i, end_i

    def fill_between(self, x=None, y1=None, y2=0, **kwargs):
        """
        Imitates matplotlib.axes.Axes.fill_between

        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.fill_between.html

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

        ekw, brush = self._setup_fill_between_colors(**kwargs)

        start_i, end_i = self._setup_fill_between_where(x, **kwargs)

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

    @staticmethod
    def _check_set_lim_kw(**kw):
        """Issues warnings for unused keywords and pops them out of kw"""
        if not kw.pop('emit', True):
            warnings.warn('emit keyword to set_xlim/set_ylim is not handled yet')
        if kw.pop('auto', False):
            warnings.warn('auto keyword to set_xlim/set_ylim is not handled yet')
        if len(kw.keys()):
            warnings.warn('set_xlim/set_ylim ignores any extra keywords in **kw: {}'.format(kw.keys()))

    @staticmethod
    def _interpret_lims(a, b):
        """Interprets the first two arguments/keywords which give limits in some form or other"""
        c = None
        if b is None and len(np.atleast_1d(a)) == 2:
            c = tuple(a)  # X limits were passed in as first argument
        elif b is not None and a is not None \
                and len(np.atleast_1d(a)) == 1 and len(np.atleast_1d(b)) == 1:
            c = (a, b)
        return c

    def set_xlim(self, left=None, right=None, **kw):
        """Direct imitation of matplotlib set_xlim"""
        new_xlims = self._interpret_lims(left, right)
        self._check_set_lim_kw(**kw)
        if new_xlims is not None:
            self.setXRange(new_xlims[0], new_xlims[1])
        return new_xlims

    def set_ylim(self, bottom=None, top=None, **kw):
        """Direct imitation of matplotlib set_ylim"""
        new_ylims = self._interpret_lims(bottom, top)
        self._check_set_lim_kw(**kw)
        if new_ylims is not None:
            self.setYRange(new_ylims[0], new_ylims[1])
        return new_ylims

    def set_xscale(self, value, **kwargs):
        """
        Imitates matplotlib set_xscale

        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xscale.html

        :param value: str
            'linear', 'log, (supported)
            'symlog', 'logit' (not yet supported by pgmpl but vaid in mpl)

        :param kwargs: additional keywords are accepted but ignored
        """
        if value == 'linear':
            self.setLogMode(x=False)
        elif value == 'log':
            self.setLogMode(x=True)
        elif value in ['symlog', 'logit']:
            warnings.warn('symlog and logistic transform scalings are not supported')
        else:
            warnings.warn('Unrecognized scale value received by set_xscale: {}. '
                          'Please try again with "linear" or "log".'.format(value))
        if len(kwargs.keys()):
            warnings.warn('Keywords to set_xscale were ignored.')

    def set_yscale(self, value, **kwargs):
        """
        Imitates matplotlib set_yscale

        https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_yscale.html

        :param value: str
            'linear', 'log, (supported)
            'symlog', 'logit' (not yet supported by pgmpl but vaid in mpl)

        :param kwargs: additional keywords are accepted but ignored
        """
        if value == 'linear':
            self.setLogMode(y=False)
        elif value == 'log':
            self.setLogMode(y=True)
        elif value in ['symlog', 'logit']:
            warnings.warn('symlog and logistic transform scalings are not supported')
        else:
            warnings.warn('Unrecognized scale value received by set_yscale: {}. '
                          'Please try again with "linear" or "log".'.format(value))
        if len(kwargs.keys()):
            warnings.warn('Keywords to set_yscale were ignored.')


class AxesImage(pg.ImageItem):
    """
    Imitates matplotlib.image.AxesImage

    https://matplotlib.org/3.1.1/api/image_api.html#matplotlib.image.AxesImage
    """

    def __init__(self, x=None, **kwargs):
        data = kwargs.pop('data', None)
        self.cmap = kwargs.pop('cmap', None)
        self.norm = kwargs.pop('norm', None)
        self.alpha = kwargs.pop('alpha', None)

        if data is not None:
            x = data['x']
            if len(data.keys()) > 1:
                warnings.warn('Axes.imshow does not extract keywords from data yet (just x).')

        self.vmin = kwargs.pop('vmin', x.min())
        self.vmax = kwargs.pop('vmax', x.max())
        self.check_inputs(**kwargs)
        self._set_up_imange_extent(x=copy.copy(x), **kwargs)

    def _set_up_imange_extent(self, x, **kwargs):
        """
        Handles setup of image extent, translate, and scale
        """
        origin = kwargs.pop('origin', None)

        if origin in ['upper', None]:
            x = x[::-1]
            extent = kwargs.pop('extent', None) or (-0.5, x.shape[1]-0.5, -(x.shape[0]-0.5), -(0-0.5))
        else:
            extent = kwargs.pop('extent', None) or (-0.5, x.shape[1]-0.5, -0.5, x.shape[0]-0.5)

        if len(np.shape(x)) == 3:
            x = np.transpose(x, (2, 0, 1))
        else:
            x = np.array(color_map_translator(
                x.flatten(),
                cmap=self.cmap,
                norm=self.norm,
                vmin=self.vmin,
                vmax=self.vmax,
                clip=kwargs.pop('clip', False),
                ncol=kwargs.pop('N', 256),
                alpha=self.alpha,
            )).T.reshape([4] + tolist(x.shape))

        super(AxesImage, self).__init__(np.transpose(x))
        self.resetTransform()
        self.translate(extent[0], extent[2])
        self.scale((extent[1] - extent[0]) / self.width(), (extent[3] - extent[2]) / self.height())

    @staticmethod
    def check_inputs(**kw):
        """Checks inputs and issues warnings as applicable"""
        if kw.pop('shape', None) is not None:
            warnings.warn('Axes.imshow ignored keyword: shape. I could not get this working with matplotlib, '
                          'so I had nothing to emulate.')
        ignoreds = ['imlim', 'resample', 'url', 'interpolation']
        for ignored in ignoreds:
            if kw.pop(ignored, None) is not None:
                warnings.warn('Axes.imshow ignored keyword: {}.'.format(ignored))
        if kw.pop('filternorm', 1) != 1 or kw.pop('filterrad', 4.0) != 4.0:
            warnings.warn('Axes.imshow ignores changes to keywords filternorm and filterrad.')
        if len(kw.keys()):
            warnings.warn('Axes.imshow got unhandled keywords: {}'.format(kw.keys()))
