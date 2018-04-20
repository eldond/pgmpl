#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Utilities for translating Matplotlib style keywords into PyQtGraph keywords
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings

# Calculation imports
import numpy as np

# Plotting imports
from PyQt4 import QtCore
import pyqtgraph as pg
try:
    from matplotlib.colors import to_rgba
except ImportError:  # Older Matplotlib versions were organized differently
    from matplotlib.colors.colorConverter import to_rgba


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
