#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import unittest
import sys

# Calculation imports
import numpy as np

# Plotting imports
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from matplotlib.colors import to_rgba


def color_translator(**kw):
    if 'color' in kw:
        if 'alpha' in kw:
            new_color = np.array(to_rgba(kw['color'])) * 255
            new_color[3] = kw['alpha'] * 255
        else:
            new_color = kw['color']
    else:
        if 'alpha' in kw:
            new_color = (0, 0, 0, int(round(kw['alpha'] * 255)))
        else:
            new_color = None
    return new_color


def style_translator(**kw):
    if 'linestyle' in kw.keys():
        style = {
            '-': QtCore.Qt.SolidLine,
            '--': QtCore.Qt.DashLine,
            '-.': QtCore.Qt.DashDotLine,
            ':': QtCore.Qt.DotLine,
            ' ': QtCore.Qt.NoPen,
            '-..': QtCore.Qt.DashDotDotLine,  # Warning: this one has no mpl equivalent: avoid it
        }.get(kw['linestyle'], None)
    else:
        style = None
    return style


def symbol_translator(**kw):
    if 'marker' in kw:
        theta = np.linspace(0, 2 * np.pi, 36)
        symbol = {  # mpl symbol : pyqt4 symbol
            '.': pg.arrayToQPath(np.cos(theta) * 0.125, np.sin(theta) * 0.125, connect='all'),
            ',': pg.arrayToQPath(np.array([-0.01, 0, 0.01, 0, -0.01]),
                                 np.array([0, 0.01, 0, -0.01, 0]), connect='all'),
            'x': pg.arrayToQPath(np.array([-0.5, 0.5, 0, 0.5, -0.5, 0]),
                                 np.array([-0.5, 0.5, 0, -0.5, 0.5, 0]), connect='all'),
            '+': '+',
            '*': 'star',
            'o': 'o',
            'v': 't',
            '^': 't1',
            '>': 't2',
            '<': 't3',
            'd': 'd',
            's': 's',
            'p': 'p',
            'h': 'h',
        }.get(kw['marker'], 'o')
    else:
        symbol = None

    return symbol


def setup_pen_kw(**kw):
    penkw = {}

    # Move the easy keywords over directly
    direct_translations_pen = {  # plotkw: pgkw
        'lw': 'width',
    }
    for direct in direct_translations_pen:
        if direct in kw:
            penkw[direct] = kw[direct_translations_pen[direct]]

    # Handle colors
    newc = color_translator(**kw)
    if newc is not None:
        penkw['color'] = newc  # If no color information was defined, leave this alone to allow default colors

    # Line style
    news = style_translator(**kw)
    if news is not None:
        penkw['style'] = news

    if len(penkw.keys()):
        pen = pg.mkPen(**penkw)
    else:
        pen = None

    return pen


def plotkw_translator(**plotkw):
    """
    Translates matplotlib plot keyword dictionary into a keyword dictionary suitable for pyqtgraph plot functions
    :param plotkw: dict
        Dictionary of matplotlib plot() keywords
    :return: dict
        Dictionary of pyqtgraph plot keywords
    """

    pgkw = {}

    # First define the pen -----------------------------------------------------------------------------------------
    pen = setup_pen_kw(**plotkw)
    if pen is not None:
        pgkw['pen'] = pen

    # Next, translate symbol related keywords ----------------------------------------------------------------------

    direct_translations = {  # pgkw: plotkw
        'symbolSize': 'markersize',
    }
    for direct in direct_translations:
        if direct in plotkw:
            pgkw[direct] = plotkw[direct_translations[direct]]

    symbol = symbol_translator(**plotkw)
    if symbol is not None:
        pgkw['symbol'] = symbol

        # Handle symbol edge
        default_mec = plotkw.get('color', None) if plotkw.get('marker', None) in ['x', '+', '.', ','] else None
        mec = plotkw.get('markeredgecolor', plotkw.get('mec', default_mec))
        mew = plotkw.get('markeredgewidth', plotkw.get('mew', None))
        penkw = {}

        if mec is not None:
            penkw['color'] = mec
        if mew is not None:
            penkw['width'] = mew
        if 'alpha' in plotkw:
            penkw['alpha'] = plotkw['alpha']
        if len(penkw.keys()):
            pgkw['symbolPen'] = setup_pen_kw(**penkw)

        # Handle fill
        brushkw = {}
        brush_color = color_translator(**plotkw)
        if brush_color is not None:
            brushkw['color'] = brush_color
        if len(brushkw.keys()):
            pgkw['symbolBrush'] = pg.mkBrush(**brushkw)

    return pgkw


def figure(*args, **kwargs):
    """
    Imitates matplotlib.pyplot.figure, but creates a PyQtGraph window with the pyqtmpl.Figure class
    :return: pyqtmpl.Figure instance
    """
    return Figure(*args, **kwargs)


class Figure:
    """
    Imitates matplotlib.figure.Figure using PyQtGraph
    """
    def __init__(
            self, figsize=None, dpi=None,
            facecolor=None, edgecolor=None, linewidth=0.0,
            frameon=None, subplotpars=None, tight_layout=None, constrained_layout=None,
    ):
        pg.setConfigOption('background', 'w' if facecolor is None else facecolor)
        pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsWindow()
        if figsize is None:
            figsize = (800, 600)
        self.win.resize(figsize[0], figsize[1])
        self.axes = None
        self.layout = pg.GraphicsLayout()
        self.win.setCentralItem(self.layout)


    def add_subplot(self, nrows, ncols, index, projection=None, polar=None, **kwargs):

        row = int(np.floor(index/ncols))
        if row > (nrows-1):
            raise ValueError('index {} would be on row {}, but the last row is {}!'.format(index, row, nrows-1))
        col = index % ncols
        # ax = self.layout.addPlot(row, col)
        ax = axes(self, row, col)
        return ax


class Axes:
    """
    Imitates matplotlib.axes.Axes using PyQtGraph
    """
    def __init__(self, fig=None, *args, **kwargs):
        if fig is None:
            fig = figure()
        self.fig = fig
        self.axes = self.fig.layout.addPlot(*args, **kwargs)

    def plot(self, *args, **kwargs):
        self.axes.plot(*args, **plotkw_translator(**kwargs))

    def set_xlabel(self, label):
        self.axes.setLabel('bottom', text=label)

    def set_ylabel(self, label):
        self.axes.setLabel('left', text=label)


def axes(*args, **kwargs):
    return Axes(*args, **kwargs)


def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
    """
    Imitates matplotlib.pyplot.subplots() using PyQtGraph
    :param nrows: int, optional, default: 1
    :param ncols: int, optional, default: 1
    :param sharex: bool or {‘none’, ‘all’, ‘row’, ‘col’}, default: False
    :param sharey: bool or {‘none’, ‘all’, ‘row’, ‘col’}, default: False
    :param squeeze: bool, optional, default: True
    :param subplot_kw: dict, optional
    :param gridspec_kw: dict, optional
    :param fig_kw:
    :return: Figure object
    :return: Axes object or array of Axes objects
    """

    def pick_share(share, ii, jj, axs_):
        if ii == 0 and jj == 0:
            return None
        if share in ['all', True]:
            return axs_[0, 0].axes
        elif share in ['col']:
            return axs_[0, jj].axes if ii > 0 else None
        elif share in ['row']:
            return axs_[ii, 0].axes if jj > 0 else None
        else:
            return None

    fig = figure(**fig_kw)
    axs = np.zeros((nrows, ncols), object)
    subplot_kw = subplot_kw if subplot_kw is not None else {}
    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols + j
            axs[i, j] = fig.add_subplot(nrows, ncols, index, **subplot_kw)
            x_share_from = pick_share(sharex, i, j, axs)
            y_share_from = pick_share(sharey, i, j, axs)
            print('index {}, row {}, col {}, xshare = {}, yshare = {}'.format(index, i, j, x_share_from, y_share_from))
            if x_share_from is not None:
                axs[i, j].axes.setXLink(x_share_from)
            if y_share_from is not None:
                axs[i, j].axes.setYLink(y_share_from)
    if squeeze:
        axs = np.squeeze(axs)
    return fig, axs


def demo_plot():
    x = np.linspace(0, 10, 151)
    y1 = x**2 + 1
    y2 = x*10 - 0.1 * x**3 + 50
    y3 = 85 - y1
    fig, axs = subplots(3, 2, sharex='col', sharey='row')
    axs[-1, 0].set_xlabel('x')
    axs[-1, 1].set_xlabel('X')
    axs[0, 0].set_ylabel('y')
    axs[1, 0].set_ylabel('y')
    axs[2, 0].set_ylabel('y')

    axs[0, 0].plot(x, y1)
    axs[0, 1].plot(x, y2)
    axs[0, 1].plot(x, y3)

    axs[1, 0].plot(x, y1, color='r')
    axs[1, 0].plot(x, y2, color='k')
    axs[1, 1].plot(x, y2, linestyle='--', color='g')
    axs[1, 1].plot(x, y3, linestyle='-.', color='b')

    axs[2, 0].plot(x, y1, color='m', marker='o')
    axs[2, 1].plot(x, y2, linestyle=' ', color='k', marker='+')
    axs[2, 1].plot(x, y3, linestyle=' ', color='k', marker='x')

    return fig, axs


class TestPyQtMpl(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest pyqtmpl
    """

    verbose = True

    plot_kw_tests = [
        {'color': 'r'},
        {'color': 'r', 'alpha': 0.5},
        {'color': 'b', 'linestyle': '--', 'marker': 'x'},
        {'color': 'g', 'linestyle': ':', 'marker': 'o'},
        {'color': 'm', 'linestyle': ' ', 'marker': 'd', 'mew': 2},
    ]

    nt = len(plot_kw_tests)

    if verbose:
        print('-' * 79)
        print('\nTestPyQtMpl has {} test sets of plot keywords ready to go!\n'.format(nt))
        print('-' * 79)

    def test_color_translator(self):
        newc = [None] * self.nt
        for i in range(self.nt):
            newc[i] = color_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New colors:', newc)

    def test_style_translator(self):
        news = [None] * self.nt
        for i in range(self.nt):
            news[i] = style_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New styles:', news)
            print('QtCore.Qt.DashDotLine = {}'.format(QtCore.Qt.DashDotLine))
            print('style_translator(linestyle="-.") = {}'.format(style_translator(linestyle="-.")))
        assert style_translator(linestyle="-.") == QtCore.Qt.DashDotLine

    def test_symbol_translator(self):
        news = [None] * self.nt
        for i in range(self.nt):
            news[i] = symbol_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New symbols:', news)

    def test_setup_pen_kw(self):
        newp = [None] * self.nt
        for i in range(self.nt):
            newp[i] = setup_pen_kw(**self.plot_kw_tests[i])
        if self.verbose:
            print('New pens:', newp)

    def test_plotkw_translator(self):
        newk = [None] * self.nt
        for i in range(self.nt):
            newk[i] = plotkw_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New keyword dictionaries:', newk)

    def test_figure(self):
        fig = figure()
        if self.verbose:
            print('test_figure: fig = {}'.format(fig))

    def test_subplots(self):
        x = np.linspace(0, 1.2, 20)
        y = x**2 + 1
        fig, axs = subplots(3, 2, sharex=True, sharey=True)
        axs[1, 1].plot(x, y)


if __name__ == '__main__':
    unittest.main()
