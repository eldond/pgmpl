#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.pyplot but using PyQtGraph to make the plots
"""

# Basic imports
from __future__ import print_function, division

# Calculation imports
import numpy as np

# pgmpl
from tracking import tracker
from figure import Figure
from axes import Axes
from util import printd


def figure(*args, **kwargs):
    """
    Imitates matplotlib.pyplot.figure, but creates a PyQtGraph window with the pgmpl.Figure class
    :return: pgmpl.Figure instance
    """
    return Figure(*args, **kwargs)


def axes(fig=None, **kwargs):
    if fig is None:
        fig = gcf()
    ax = fig.add_subplot(1, 1, 1, **kwargs)
    return ax


def subplots(nrows=1, ncols=1, **fig_kw):
    """
    Imitates matplotlib.pyplot.subplots() using PyQtGraph
    :param nrows: int, optional, default: 1
    :param ncols: int, optional, default: 1
    :param sharex: bool or {‘none’, ‘all’, ‘row’, ‘col’}, default: False
    :param sharey: bool or {‘none’, ‘all’, ‘row’, ‘col’}, default: False
    :param squeeze: bool, optional, default: True
    :param subplot_kw: dict, optional
    :param gridspec_kw: dict, optional
    :param fig_kw: Remaining keywords are passed to pyplot.figure() call to create figure.Figure instance.
    :return: Figure object
    :return: Axes object or array of Axes objects
    """
    gridspec_kw = fig_kw.pop('gridspec_kw', None)

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
    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols + j + 1
            axs[i, j] = fig.add_subplot(
                nrows, ncols, index,
                sharex=pick_share(fig_kw.pop('sharex', 'none'), i, j, axs),
                sharey=pick_share(fig_kw.pop('sharey', 'none'), i, j, axs),
                **fig_kw.pop('subplot_kw', {}))
            printd('index {}, row {}, col {}'.format(index, i, j))
    if fig_kw.pop('squeeze', True):
        axs = np.squeeze(axs)
        if len(np.shape(axs)) == 0:
            axs = axs[()]  # https://stackoverflow.com/a/35160426/6605826
    return fig, axs


def subplots_adjust(**kwargs):
    """Imitates matplotlib.pyplot.subplots_adjust. Also accepts fig keyword."""
    from matplotlib.figure import SubplotParams
    fig = kwargs.pop('fig', gcf())
    fig.set_subplotpars(SubplotParams(**kwargs))


def suptitle(t, **kwargs):
    gcf().suptitle(t, **kwargs)


def gcf():
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


def close(fig=None):
    if fig is None:
        fig = gcf()
    try:
        fig.close()
    except AttributeError:
        # pyqtgraph.PlotWidget.close() fails every other time it's called, so double tap for the win.
        fig.close()
