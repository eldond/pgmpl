#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.pyplot but using PyQtGraph to make the plots
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings

# Calculation imports
import numpy as np

# pyqtmpl
from tracking import tracker
from figure import Figure
from axes import Axes
from util import printd


def figure(*args, **kwargs):
    """
    Imitates matplotlib.pyplot.figure, but creates a PyQtGraph window with the pyqtmpl.Figure class
    :return: pyqtmpl.Figure instance
    """
    return Figure(*args, **kwargs)


def axes(fig=None, **kwargs):
    if fig is None:
        fig = gcf()
    ax = fig.add_subplot(1, 1, 0, **kwargs)
    return ax


def subplots(nrows=1, ncols=1, sharex='none', sharey='none', squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
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
            return axs_[0, 0]
        elif share in ['col']:
            return axs_[0, jj] if ii > 0 else None
        elif share in ['row']:
            return axs_[ii, 0] if jj > 0 else None
        else:
            return None

    fig = figure(**fig_kw)

    if gridspec_kw is not None:
        gridkw = ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']
        if any([thing in gridspec_kw.keys() for thing in gridkw]):
            from matplotlib.figure import SubplotParams
            spkw = {}
            for thing in gridkw:
                spkw[thing] = gridspec_kw.pop(thing, None)
            sp = SubplotParams(**spkw)
            fig.set_subplotpars(sp)

    axs = np.zeros((nrows, ncols), object)
    subplot_kw = subplot_kw if subplot_kw is not None else {}
    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols + j
            axs[i, j] = fig.add_subplot(
                nrows, ncols, index,
                sharex=pick_share(sharex, i, j, axs), sharey=pick_share(sharey, i, j, axs),
                **subplot_kw)
            printd('index {}, row {}, col {}'.format(index, i, j))
    if squeeze:
        axs = np.squeeze(axs)
        if len(np.shape(axs)) == 0:
            axs = axs[()]  # https://stackoverflow.com/a/35160426/6605826
    return fig, axs


def gcf():
    # Improvement may be possible. See:
    # QWidget *QApplication::activeWindow()  http://doc.qt.io/qt-5/qapplication.html#activeWindow
    if len(tracker.open_windows):
        return tracker.open_windows[-1]
    else:
        return figure()


def gca(fig=None):
    if fig is None:
        fig = gcf()
    return fig.gca()


def plot(*args, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        fig = kwargs.pop('fig', None)
        if fig is None:
            fig = gcf()
        ax = fig.gca()
    ax.plot(*args, **kwargs)
