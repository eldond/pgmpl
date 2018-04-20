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
from figure import Figure
from axes import Axes
from util import printd


def figure(*args, **kwargs):
    """
    Imitates matplotlib.pyplot.figure, but creates a PyQtGraph window with the pyqtmpl.Figure class
    :return: pyqtmpl.Figure instance
    """
    return Figure(*args, **kwargs)


def axes(**kwargs):
    return Axes(**kwargs)


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
            axs[i, j] = fig.add_subplot(nrows, ncols, index, **subplot_kw)
            x_share_from = pick_share(sharex, i, j, axs)
            y_share_from = pick_share(sharey, i, j, axs)
            printd('index {}, row {}, col {}, xshare = {}, yshare = {}'.format(index, i, j, x_share_from, y_share_from))
            if x_share_from is not None:
                axs[i, j].setXLink(x_share_from)
            if y_share_from is not None:
                axs[i, j].setYLink(y_share_from)
    if squeeze:
        axs = np.squeeze(axs)
        if len(np.shape(axs)) == 0:
            axs = axs[()]  # https://stackoverflow.com/a/35160426/6605826
    return fig, axs
