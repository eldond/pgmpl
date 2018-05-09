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
import unittest

# Calculation imports
import numpy as np

# Plotting imports
from PyQt4 import QtCore
import pyqtgraph as pg
import matplotlib.cm
from matplotlib import rcParams
from matplotlib.colors import Normalize
try:
    from matplotlib.colors import to_rgba
except ImportError:  # Older Matplotlib versions were organized differently
    from matplotlib.colors import colorConverter
    to_rgba = colorConverter.to_rgba

# pgmpl imports
from util import set_debug, printd, tolist


def dealias(**kws):
    missing_value_mark = 'This value is missing; it is not just None. We want to allow for the possibility that ' \
                         'keyword=None is treated differently than a missing keyword in some function, because a ' \
                         'function might not accept unrecognized keywords.'
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
    for primary, aliases in alias_lists.iteritems():
        aliasv = {alias: kws.pop(alias, missing_value_mark) for alias in aliases}
        not_missing = [v != missing_value_mark for v in aliasv.values()]
        if primary not in kws.keys() and any(not_missing):
            # The aliases only need be considered if the primary is missing.
            aliasu = np.atleast_1d(aliasv.keys())[np.atleast_1d(not_missing)][0]
            kws[primary] = aliasv[aliasu]
            printd("  assigned kws['{}'] = kws.pop('{}')".format(primary, aliasu))
        else:
            printd(' did not asssign {}'.format(primary))
    return kws


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
        'lines.linestyle': 'linestyle',  # This could've gone in simples, but then there'd be no example.
    }
    simples = [
        'lines.linewidth',
        'lines.marker',
        'lines.markeredgewidth',
        'lines.markersize',
    ]
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
            new_color = (0, 0, 0, 0)  # Empty strings and spaces are code for invisible (alpha = 0)
        elif 'alpha' in kw and kw['alpha'] is not None:
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


def color_map_translator(x, cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256, alpha=None):
    """
    Translates colors for a matplotlib colormap and a dataset, such as would be used for scatter, imshow, contour, etc.
    :param x: numeric scalar or iterable
        Data to be mapped. Very boring if scalar.
    :param cmap: string
    :param norm: matplotlib normalization class. Defaults to new instance of mpl.colors.Normalize if None
    :param vmin: number: lower limit passed to Normalize if norm is None
    :param vmax: number: lower limit passed to Normalize if norm is None
    :param clip: bool
    :param ncol: int: passed to Colormap to set number of colors
    :param alpha: float: opacity from 0 to 1 or None
    :return: List of pyqtgraph-compatible color specifications with length matching x
    """
    printd('color_map_translator...')
    if norm is None:
        printd('  norm was None, normalizing...')
        norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)
    comap = matplotlib.cm.get_cmap(cmap, lut=ncol)
    colors = comap(norm(np.atleast_1d(x)))
    return [color_translator(color=color, alpha=alpha) for color in tolist(colors)]


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
            'None': None,
            'none': None,
            None: None,
        }.get(kw['marker'], 'o')
    else:
        symbol = None

    return symbol


def setup_pen_kw(**kw):
    """
    Builds a pyqtgraph pen (object containing color, linestyle, etc. information) from Matplotlib keywords.
    Please dealias first.

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
        'linewidth': 'width',
    }
    for direct in direct_translations_pen:
        if direct in kw:
            penkw[direct_translations_pen[direct]] = kw[direct]

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

    # Pass through other keywords
    late_pops = ['color', 'alpha', 'linewidth', 'marker', 'linestyle']
    for late_pop in late_pops:
        # Didn't pop yet because used in a few places or popping above is inside of an if and may not have happened
        plotkw.pop(late_pop, None)
    plotkw.update(pgkw)

    printd('plotkw symbol = {}; symbol = {}'.format(plotkw.get('symbol', 'no symbol defined'), symbol), level=1)
    return plotkw


class TestPgmplTranslate(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest translate
    """

    verbose = False

    plot_kw_tests = [
        {'color': 'r'},
        {'color': 'r', 'alpha': 0.5},
        {'color': 'b', 'linestyle': '--', 'marker': 'x'},
        {'color': 'g', 'linestyle': ':', 'marker': 'o'},
        {'color': 'm', 'linestyle': ' ', 'marker': 'd', 'mew': 2},
        {'color': ' ', 'markeredgewidth': 1.5},
    ]

    nt = len(plot_kw_tests)

    if verbose:
        print('-' * 79)
        print('\nTestPgmpl has {} test sets of plot keywords ready to go!\n'.format(nt))
        print('-' * 79)
        set_debug(True)

    def test_defaults_from_rcparams(self):
        unique_linewidth = 5.1234958293
        ans0 = defaults_from_rcparams({})
        ans1 = defaults_from_rcparams({'linewidth': unique_linewidth})
        ans0b = copy.deepcopy(ans0)
        ans0b['linewidth'] = unique_linewidth
        assert ans1 == ans0b
        if rcParams['lines.linewidth'] == unique_linewidth:
            assert ans1 == ans0
        else:
            assert ans1 != ans0

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
        newk = [{}] * self.nt
        for i in range(self.nt):
            newk[i] = plotkw_translator(**self.plot_kw_tests[i])
        if self.verbose:
            print('New keyword dictionaries:', newk)

    def test_dealias(self):
        test_dict = {'lw': 5, 'ls': '--', 'mec': 'r', 'markeredgewidth': 1, 'blah': 0}
        correct_answer = {'linewidth': 5, 'linestyle': '--', 'markeredgecolor': 'r', 'markeredgewidth': 1, 'blah': 0}
        test_answer = dealias(**test_dict)
        assert correct_answer == test_answer  # https://stackoverflow.com/a/5635309/6605826
        if self.verbose:
            print('test_dealias: test_answer = {}'.format(test_answer))

    def test_color_map_translator(self):
        x = [0, 1, 2, 3, 5, 9, 10, 22]
        m1 = color_map_translator(1.579, cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256, alpha=0.5)
        m2 = color_map_translator([1, 2, 3], cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256)
        m3 = color_map_translator(x, cmap=None, norm=None, vmin=None, vmax=None, clip=False, ncol=256)
        m4 = color_map_translator(x, cmap='plasma', norm=None, vmin=None, vmax=None, clip=False, ncol=256)
        assert len(m1) == 1
        assert len(m2) == 3
        assert len(m3) == len(x)
        assert any((np.atleast_1d(m3) != np.atleast_1d(m4)).flatten())
        if self.verbose:
            print('test_color_map_translator: m4 = {}'.format(m4))


if __name__ == '__main__':
    unittest.main()
