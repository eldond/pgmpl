#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Contains tools for helping set up matplotlib keywords, defaults, and drawing objects, like PyQt pens.

Also has the more complicated plot keyword translator that uses basic translations and the tools.
"""

# Basic imports
from __future__ import print_function, division
import os
import numpy as np
import copy

# Plotting imports
import pyqtgraph as pg
from matplotlib import rcParams

# pgmpl imports
from pgmpl.util import printd
from pgmpl.translate import color_translator, style_translator, symbol_translator


def dealias(**kws):
    """
    Checks for alias of a keyword (like 'ls' for linestyle) and updates keywords so that the primary is defined.

    That is, if kws contains 'ls', the result will contain 'linestyle' with the value defined by 'ls'. This
    eliminates the need to check through all the aliases later; other functions can rely on finding the primary
    keyword. Also, the aliases are all removed so they don't confuse other functions.

    :param kws: keywords to dealias

    :return: dict
        Dictionary with all aliases replaced by primary keywords (ls is replaced by linestyle, for example)
    """
    alias_lists = {  # If there is more than one alias, then the first one in the list is used
        'linewidth': ['lw'],
        'linestyle': ['ls'],
        'markeredgewith': ['mew'],
        'markeredgecolor': ['mec'],
        'markerfacecolor': ['mfc'],
        'markersize': ['ms'],
        'markerfacecoloralt': ['mfcalt'],
        'antialiased': ['aa'],
        'color': ['c'],
        'edgecolor': ['ec'],
        'facecolor': ['fc'],
        'verticalalignment': ['va'],
        'horizontalalignment': ['ha'],
    }
    for primary, aliases in list(alias_lists.items()):  # https://stackoverflow.com/a/13998534/6605826
        for alias in aliases:
            if alias in kws:
                if primary not in kws.keys():
                    kws[primary] = kws.pop(alias)
                    printd("  assigned kws['{}'] = kws.pop('{}')".format(primary, alias))
                else:
                    kws.pop(alias)
                    printd(' did not asssign {}'.format(primary))
    return kws


def symbol_edge_setup(pgkw, plotkw):
    """
    Manage keywords related to symbol edges

    :param pgkw: Dictionary of new keywords to pass to pyqtgraph functions

    :param plotkw: Dictionary of matplotlib style keywords (translation in progress)
    """
    default_mec = plotkw.get('color', None) if plotkw.get('marker', '') in ['x', '+', '.', ',', '|', '_'] else None
    mec = plotkw.pop('markeredgecolor', default_mec)
    mew = plotkw.pop('markeredgewidth', None)
    symbol = symbol_translator(**plotkw)
    if symbol is not None:
        pgkw['symbol'] = symbol
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
    else:
        pgkw.pop('symbolSize', None)  # This isn't used when symbol is undefined, but it can cause problems, so remove.
    printd('plotkw symbol = {}; symbol = {}'.format(plotkw.get('symbol', 'no symbol defined'), symbol), level=1)


def setup_pen_kw(penkw={}, **kw):
    """
    Builds a pyqtgraph pen (object containing color, linestyle, etc. information) from Matplotlib keywords.
    Please dealias first.

    :param penkw: dict
        Dictionary of pre-translated pyqtgraph keywords to pass to pen

    :param kw: dict
        Dictionary of Matplotlib style plot keywords in which line plot relevant settings may be specified. The entire
        set of mpl plot keywords may be passed in, although only the keywords related to displaying line plots will be
        used here.

    :return: pyqtgraph pen instance
        A pen which can be input with the pen keyword to many pyqtgraph functions
    """

    # Move the easy keywords over directly
    direct_translations_pen = {  # plotkw: pgkw
        'linewidth': 'width',
    }
    for direct in direct_translations_pen:
        penkw[direct_translations_pen[direct]] = kw.pop(direct, None)

    # Handle colors and styles
    penkw['color'] = color_translator(**kw)
    penkw['style'] = style_translator(**kw)

    # Prune values of None
    penkw = {k: v for k, v in penkw.items() if v is not None}

    return pg.mkPen(**penkw) if len(penkw.keys()) else None


def defaults_from_rcparams(plotkw):
    """
    Given a dictionary of Matplotlib style plotting keywords, any missing keywords (color, linestyle, etc.) will be
    added using defaults determined by Matplotlib's rcParams. Please dealias plotkw first.

    :param plotkw: dict
         Dictionary of Matplotlib style plot keywords

    :return: dict
        Input dictionary with missing keywords filled in using defaults
    """
    params = {  # If you have a parameter that can't be assigned simply by just splitting after ., then set it up here.
        # 'lines.linestyle': 'linestyle',  # This can go in simples instead, but here's how it would go for example.
    }
    simples = ['lines.linewidth', 'lines.marker', 'lines.markeredgewidth', 'lines.markersize', 'lines.linestyle']
    for simple in simples:
        params[simple] = '.'.join(simple.split('.')[1:])

    for param in params.keys():
        if not params[param] in plotkw.keys():
            # Keyword is missing
            plotkw[params[param]] = rcParams[param]
            printd("  assigned plotkw['{}'] = rcParams[{}] = {}".format(params[param], param, rcParams[param]),
                   level=2)
        else:
            printd("  keywords {} exists in plotkw; no need to assign default from rcParams['{}']".format(
                params[param], param), level=2)

    return plotkw


def plotkw_translator(**plotkw):
    """
    Translates matplotlib plot keyword dictionary into a keyword dictionary suitable for pyqtgraph plot functions

    :param plotkw: dict
        Dictionary of matplotlib plot() keywords

    :return: dict
        Dictionary of pyqtgraph plot keywords
    """

    pgkw = {}
    plotkw = dealias(**copy.deepcopy(plotkw))  # Copy: Don't break the original in case it's needed for other calls
    plotkw = defaults_from_rcparams(plotkw)

    # First define the pen -----------------------------------------------------------------------------------------
    pen = setup_pen_kw(**plotkw)
    if pen is not None:
        pgkw['pen'] = pen

    # Next, translate symbol related keywords ----------------------------------------------------------------------

    direct_translations = {  # mpl style plotkw: pg style pgkw
        'markersize': 'symbolSize',
        'pg_label': 'label',  # Not a real mpl keyword, but otherwise there would be no way to access pg's label
        'label': 'name',
    }
    for direct in direct_translations:
        if direct in plotkw:
            pgkw[direct_translations[direct]] = plotkw.pop(direct)

    # Handle symbol edge
    symbol_edge_setup(pgkw, plotkw)

    # Pass through other keywords
    late_pops = ['color', 'alpha', 'linewidth', 'marker', 'linestyle']
    for late_pop in late_pops:
        # Didn't pop yet because used in a few places or popping above is inside of an if and may not have happened
        plotkw.pop(late_pop, None)
    plotkw.update(pgkw)

    return plotkw
