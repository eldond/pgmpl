#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Utilities for translating Matplotlib style keywords into PyQtGraph keywords

This file is for simple translations. Complicated or compound tools go in tools.
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import copy

# Calculation imports
import numpy as np

# Plotting imports
from pyqtgraph import QtCore
import pyqtgraph as pg
import matplotlib.cm
from matplotlib.colors import Normalize
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
try:
    from matplotlib.colors import to_rgba
except ImportError:  # Older Matplotlib versions were organized differently
    from matplotlib.colors import colorConverter
    to_rgba = colorConverter.to_rgba

# pgmpl imports
from pgmpl.util import printd, tolist


# Install custom symbols
theta = np.linspace(0, 2 * np.pi, 36)
custom_symbols = {
    '.': pg.arrayToQPath(np.cos(theta) * 0.125, np.sin(theta) * 0.125, connect='all'),
    ',': pg.arrayToQPath(np.array([-0.01, 0, 0.01, 0, -0.01]), np.array([0, 0.01, 0, -0.01, 0]), connect='all'),
    '_': pg.arrayToQPath(np.array([-0.5, 0.5]), np.array([0, 0]), connect='all'),
    '|': pg.arrayToQPath(np.array([0, 0]), np.array([-0.5, 0.5]), connect='all'),
    'x': pg.arrayToQPath(
        np.array([-0.5, 0.5, 0, 0.5, -0.5, 0]), np.array([-0.5, 0.5, 0, -0.5, 0.5, 0]), connect='all',
    ),
}
for symb in custom_symbols:
    if symb not in Symbols:
        Symbols[symb] = custom_symbols[symb]


def color_translator(**kw):
    """
    Translates colors specified in the Matplotlib system into pyqtgraph color descriptions

    :param kw: dict
        Dictionary of Matplotlib style plot keywords in which color may be specified. The entire set of mpl plot
        keywords may be passed in, although only color-relevant ones will be used.

    :return: iterable
        An RGBA color description (each from 0 to 255) for use with pyqtgraph
    """
    if 'color' in kw and kw['color'] is not None:
        try:
            printd('    color_translator input: kw["color"] = {}, to_rgba(kw.get("color", None)) = {}'.format(
                kw.get('color', None), to_rgba(kw.get('color', None))), level=3)
        except ValueError:
            printd('    color_translator input: kw["color"] = {}'.format(kw.get('color', None)), level=3)
        if kw['color'] in ['', ' ']:
            return 0, 0, 0, 0  # Empty strings and spaces are code for invisible (alpha = 0)
        elif 'alpha' in kw and kw['alpha'] is not None:
            return np.append(np.array(to_rgba(kw['color']))[0:3], kw['alpha']) * 255
        else:
            return np.array(to_rgba(kw['color'])) * 255
    else:
        return (0, 0, 0, int(round(kw['alpha'] * 255))) if 'alpha' in kw and kw['alpha'] is not None else None


def color_map_translator(x, **kw):
    """
    Translates colors for a matplotlib colormap and a dataset, such as would be used for scatter, imshow, contour, etc.

    :param x: numeric scalar or iterable
        Data to be mapped. Very boring if scalar.

    :param cmap: string
        Color map nape, passed to matplotlib.cm.get_cmap()

    :param norm: matplotlib normalization class
        Defaults to new instance of mpl.colors.Normalize

    :param vmin: numeric
        Lower limit passed to new Normalize instance if norm is None; ignored if norm is provided.

    :param vmax: numeric
        Lower limit passed to new Normalize instance if norm is None; ignored if norm is provided.

    :param clip: bool
        Passed to Normalize if a new Normalize instance is created. Otherwise, not used.

    :param ncol: int
        passed to Colormap to set number of colors

    :param alpha: float:
        opacity from 0 to 1 or None

    :return: list
        List of pyqtgraph-compatible color specifications with length matching x
    """
    printd('color_map_translator...')
    norm = kw.pop('norm', None)
    if norm is None:
        printd('  norm was None, normalizing...')
        norm = Normalize(vmin=kw.pop('vmin', None), vmax=kw.pop('vmax', None), clip=kw.pop('clip', False))
    comap = matplotlib.cm.get_cmap(kw.pop('cmap', None), lut=kw.pop('ncol', 256))
    colors = comap(norm(np.atleast_1d(x)))
    return [color_translator(color=color, alpha=kw.get('alpha', None)) for color in tolist(colors)]


def style_translator(**kw):
    """
    Translates linestyle descriptions from the Matplotlib system into Qt pen styles. Please dealias first.
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
    # mpl symbol : pyqt4 symbol
    pyqt_symbol = {
        '.': '.', ',': ',', 'x': 'x', '+': '+', '*': 'star', 'o': 'o', 'v': 't', '^': 't1', '>': 't2', '<': 't3',
        'd': 'd', 's': 's', 'p': 'p', 'h': 'h', '_': '_', '|': '|', 'None': None, 'none': None, None: None,
    }.get(kw.get('marker', None), 'o')
    return pyqt_symbol
