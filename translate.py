#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Utilities for translating Matplotlib style keywords into PyQtGraph keywords
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import copy

# Calculation imports
import numpy as np

# Plotting imports
from PyQt4 import QtCore
import pyqtgraph as pg
try:
    from matplotlib.colors import to_rgba
except ImportError:  # Older Matplotlib versions were organized differently
    from matplotlib.colors import colorConverter
    to_rgba = colorConverter.to_rgba

# pyqtmpl imports
from util import printd


def color_translator(**kw):
    """
    Translates colors specified in the Matplotlib system into pyqtgraph color descriptions
    :param kw: dict
        Dictionary of Matplotlib style plot keywords in which color may be specified. The entire set of mpl plot
        keywords may be passed in, although only color-relevant ones will be used.
    :return: iterable
        An RGBA color description (each from 0 to 255) for use with pyqtgraph
    """
    if 'color' in kw:
        printd('    color_translator input: kw["color"] = {}, to_rgba(kw.get("color", None)) = {}'.format(
            kw.get('color', None), to_rgba(kw.get('color', None))), level=3)
        if 'alpha' in kw and kw['alpha'] is not None:
            new_color = np.array(to_rgba(kw['color'])) * 255
            new_color[3] = kw['alpha'] * 255
        else:
            new_color = np.array(to_rgba(kw['color'])) * 255
    else:
        if 'alpha' in kw and kw['alpha'] is not None:
            new_color = (0, 0, 0, int(round(kw['alpha'] * 255)))
        else:
            new_color = None
    return new_color


def style_translator(**kw):
    """
    Translates linestyle descriptions from the Matplotlib system into Qt pen styles
    :param kw: dict
        Dictionary of Matplotlib style plot keywords in which linestyle may be specified. The entire set of mpl plot
        keywords may be passed in, although only linestyle-relevant ones will be used.
    :return: A Qt pen style suitable for use in pyqtgraph.mkPen()
    """
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
    """
    Translates Matplotlib markers into pyqtgraph symbols.
    :param kw: dict
        Dictionary of Matplotlib style plot keywords in which marker may be specified. The entire set of mpl plot
        keywords may be passed in, although only linestyle-relevant ones will be used.
    :return: string
        Code for the relevant pyqtgraph symbol.
    """
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
            '_': pg.arrayToQPath(np.array([-0.5, 0.5]), np.array([0, 0]), connect='all'),
            '|': pg.arrayToQPath(np.array([0, 0]), np.array([-0.5, 0.5]), connect='all'),
        }.get(kw['marker'], 'o')
    else:
        symbol = None

    return symbol


def setup_pen_kw(**kw):
    """
    Builds a pyqtgraph pen (object containing color, linestyle, etc. information) from Matplotlib keywords
    :param kw: dict
        Dictionary of Matplotlib style plot keywords in which line plot relevant settings may be specified. The entire
        set of mpl plot keywords may be passed in, although only the keywords related to displaying line plots will be
        used here.
    :return: pyqtgraph pen instance
        A pen which can be input with the pen keyword to many pyqtgraph functions
    """
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
    plotkw = copy.deepcopy(plotkw)  # Don't break the original in case it's needed for other calls

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
            pgkw[direct] = plotkw.pop(direct_translations[direct])

    symbol = symbol_translator(**plotkw)
    if symbol is not None:
        pgkw['symbol'] = symbol

        # Handle symbol edge
        default_mec = plotkw.get('color', None) if plotkw.pop('marker', '') in ['x', '+', '.', ',', '|', '_'] else None
        mec = plotkw.pop('markeredgecolor', plotkw.pop('mec', default_mec))
        mew = plotkw.pop('markeredgewidth', plotkw.pop('mew', None))
        penkw = {}

        if mec is not None:
            penkw['color'] = mec
        if mew is not None:
            penkw['width'] = mew
        if 'alpha' in plotkw:
            penkw['alpha'] = plotkw.pop('alpha')
        if len(penkw.keys()):
            pgkw['symbolPen'] = setup_pen_kw(**penkw)

        # Handle fill
        brushkw = {}
        brush_color = color_translator(**plotkw)
        if brush_color is not None:
            brushkw['color'] = brush_color
        if len(brushkw.keys()):
            pgkw['symbolBrush'] = pg.mkBrush(**brushkw)

    # Pass through other keywords
    late_pops = ['color', 'alpha', 'lw', 'marker', 'linestyle']
    for late_pop in late_pops:
        # Didn't pop yet because used in a few places or popping above is inside of an if and may not have happened
        plotkw.pop(late_pop, None)
    plotkw.update(pgkw)

    return plotkw
