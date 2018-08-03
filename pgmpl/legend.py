#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.legend but using PyQtGraph.

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

# pgmpl
# noinspection PyUnresolvedReferences
import pgmpl.__init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.util import printd, tolist, is_numeric


class Legend:
    """
    Post-generated legend for pgmpl.axes.Axes. This is not a direct imitation of Matplotlib's Legend as it has to
    accept events from pyqtgraph. It also has to bridge the gap between Matplotlib style calling legend after plotting
    and pyqtgraph style calling legend first.

    The call method is supposed to imitate matplotlib.axes.Axes.legend(), though. Bind a Legend class instance to an
    Axes instance as Axes.legend = Legend() and then call Axes.legend() as in matplotlib.
    The binding is done in Axes class __init__.
    """
    def __init__(self, ax=None):
        """
        :param ax: Axes instance
            Reference to the plot axes to which this legend is attached (required).
        """
        # noinspection PyProtectedMember
        from pyqtgraph.graphicsItems.ViewBox.ViewBox import ChildGroup
        self.ax = ax
        # pyqtgraph legends just don't work with some items yet. Avoid errors by trying to use these classes as handles:
        self.unsupported_item_classes = [
            pg.graphicsItems.FillBetweenItem.FillBetweenItem,
            pg.InfiniteLine,
            ChildGroup,
        ]
        #   File "/lib/python2.7/site-packages/pyqtgraph/graphicsItems/LegendItem.py", line 149, in paint
        #     opts = self.item.opts
        # AttributeError: 'InfiniteLine' object has no attribute 'opts'
        self.items_added = []
        self.drag_enabled = True
        self.leg = None

    def supported(self, item):
        """Quick test for whether or not item (which is some kind of plot object) is supported by this legend class"""
        return not any([isinstance(item, uic) for uic in self.unsupported_item_classes])

    @staticmethod
    def handle_info(handles, comment=None):
        """For debugging: prints information on legend handles"""
        if comment is not None:
            printd(comment)
        for i, handle in enumerate(tolist(handles)):
            printd('  {i:02d} handle name: {name:}, class: {cls:}, isVisible: {isvis:}'.format(
                i=i,
                name=handle.name() if hasattr(handle, 'name') else None,
                cls=handle.__class__ if hasattr(handle, '__class__') else ' not found ',
                isvis=handle.isVisible() if hasattr(handle, 'isVisible') else None,
            ))

    def get_visible_handles(self):
        """
        :return: List of legend handles for visible plot items
        """
        handles = self.ax.getViewBox().allChildren()
        self.handle_info(handles, comment='handles from allChildren')
        return [item for item in handles if hasattr(item, 'isVisible') and item.isVisible()]

    @staticmethod
    def _cleanup_legend_labels(handles, labels):
        nlab = len(np.atleast_1d(labels))
        if labels is not None and nlab == 1:
            labels = tolist(labels)*len(handles)
        elif labels is not None and nlab == len(handles):
            labels = tolist(labels)
        else:
            handles = [item for item in handles if hasattr(item, 'name') and item.name() is not None]
            labels = [item.name() for item in handles]
        return handles, labels

    def __call__(self, handles=None, labels=None, **kw):
        """
        Adds a legend to the plot axes. This class should be added to axes as they are created so that calling it acts
        like a method of the class and adds a legend, imitating matplotlib legend calling.
        """
        printd('  custom legend call')
        self.leg = self.ax.addLegend()
        # ax.addLegend modifies ax.legend, so we have to put it back in order to
        # preserve a reference to pgmpl.axes.Legend.
        self.ax.legend = self

        handles = tolist(handles if handles is not None else self.get_visible_handles())

        for handle, label in zip(*self._cleanup_legend_labels(handles, labels)):
            if self.supported(handle):
                self.leg.addItem(handle, label)

        self.check_call_kw(**kw)

        return self

    @staticmethod
    def check_call_kw(**kw):
        """Checks keywords passed to Legend.__call__ and warns about unsupported ones"""
        unhandled_kws = dict(
            loc=None, numpoints=None, markerscale=None, markerfirst=True, scatterpoints=None, scatteryoffsets=None,
            prop=None, fontsize=None, borderpad=None, labelspacing=None, handlelength=None, handleheight=None,
            handletextpad=None, borderaxespad=None, columnspacing=None, ncol=1, mode=None, fancybox=None, shadow=None,
            title=None, framealpha=None, edgecolor=None, facecolor=None, bbox_to_anchor=None, bbox_transform=None,
            frameon=None, handler_map=None,
        )
        for unhandled in unhandled_kws.keys():
            if unhandled in kw.keys():
                kw.pop(unhandled)
                warnings.warn('pgmpl.axes.Legend.__call__ got unhandled keyword: {}. '
                              'This keyword might be implemented later.'.format(unhandled))
        if len(kw.keys()):
            warnings.warn('pgmpl.axes.Legend.__call__ got unrecognized keywords: {}'.format(kw.keys()))

    def addItem(self, item, name=None):
        """
        pyqtgraph calls this method of legend and so it must be provided.
        :param item: plot object instance
        :param name: string
        """
        self.items_added += [(item, name)]  # This could be used in place of the search for items, maybe.
        return None

    def draggable(self, on_off=True):
        """
        Provided for compatibility with matplotlib legends, which have this method.
        pyqtgraph legends are always draggable.
        :param on_off: bool
            Throws a warning if user attempts to disable draggability
        """
        self.drag_enabled = on_off
        if not on_off:
            warnings.warn(
                'Draggable switch is not enabled yet. The draggable() method is provided to prevent failures when '
                'plotting routines are converted from matplotlib. pyqtgraph legends are draggable by default.'
            )
        return None

    def clear(self):
        """Removes the legend from Axes instance"""
        printd('  Clearing legend {}...'.format(self.leg))
        try:
            self.leg.scene().removeItem(self.leg)  # https://stackoverflow.com/a/42794442/6605826
        except AttributeError:
            printd('  Could not clear legend (maybe it is already invisible?')
