#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.figure but using PyQtGraph to make the plots

Classes and methods imitate Matplotlib counterparts as closely as possible, so please see Matplotlib documentation for
more information.
"""

# Basic imports
from __future__ import print_function, division
import warnings

# Calculation imports
import numpy as np
from matplotlib.cbook import flatten

# Plotting imports
import pyqtgraph as pg
from matplotlib import rcParams

# pgmpl
# noinspection PyUnresolvedReferences
import pgmpl.__init__  # __init__ does setup stuff like making sure a QApp exists
from pgmpl.tracking import tracker
from pgmpl.axes import Axes
from pgmpl.util import printd, tolist
from pgmpl.colorbar import Colorbar


class Figure(pg.PlotWidget):
    """
    Imitates matplotlib.figure.Figure using PyQtGraph
    """
    def __init__(self, **kw):
        figsize = kw.pop('figsize', None)
        dpi = kw.pop('dpi', None)

        super(Figure, self).__init__()
        self.patch_resize = True  # Controls whether resize events mess with margins or not (this works well now)
        tracker.window_opened(self)
        self.tight = kw.pop('tight_layout', None) or kw.pop('constrained_layout', None)
        self.margins = {'left': 10, 'top': 10, 'right': 10, 'bottom': 10, 'hspace': 10, 'wspace': 10} \
            if self.tight else None
        self.resizeEvent_original = self.resizeEvent
        self.resizeEvent = self.resize_event
        dpi = rcParams['figure.dpi'] if dpi is None else dpi
        figsize = rcParams['figure.figsize'] if figsize is None else figsize
        self.width, self.height = (np.array(figsize)*dpi).astype(int)
        self.resize(self.width, self.height)
        for init_to_none in ['axes', 'suptitle_label']:
            setattr(self, init_to_none, None)
        self.suptitle_text = ''
        self.fig_colspan = 1
        self.layout = self.mklay()
        self.clear = self.clearfig  # Just defining the thing as clear doesn't work; needs to be done this way.

        self.set_subplotpars(kw.pop('subplotpars', None))

        if kw.pop('frameon', True) is not False and kw.pop('linewidth', 0.0) > 0 and kw.pop('edgecolor', None):
            warnings.warn('WARNING: frame around figure edge is not implemented yet')

    def clearfig(self):
        """Method for clearing the figure. Gets assigned to self.clear"""
        del self.layout
        self.suptitle_text = ''
        self.fig_colspan = 1
        self.layout = self.mklay()

    def mklay(self):
        """Method for creating layout; used in __init__ and after clear"""
        layout = pg.GraphicsLayout()
        self.setCentralItem(layout)
        self.show()
        return layout

    def resize_event(self, event):
        """
        Intercepts resize events and updates tracked height and width. Needed for keeping subplotpars up to date.
        Not a matplotlib imitation, but used for interfacing with pyqtgraph behavior.
        :param event: window resize event
        """
        if hasattr(event, 'size'):
            self.width = event.size().width()
            self.height = event.size().height()
            if self.patch_resize and not self.tight:
                self.set_subplotpars(None)
        self.resizeEvent_original(event)
        return

    def set_subplotpars(self, pars):
        """
        Sets margins and spacing between Axes. Not a direct matplotlib imitation.

        :param pars: SubplotParams instance
            The subplotpars keyword to __init__ goes straight to here.
        """
        if pars is None or self.layout is None:
            # Either no pars were provided or the layout has already been set to None because the figure is closing.
            # Don't do any margin adjustments.
            return
        self.margins = {
            'left': pars.left, 'top': pars.top, 'right': pars.right, 'bottom': pars.bottom,
            'hspace': pars.hspace, 'wspace': pars.wspace,
        }
        if self.margins is not None:
            if self.tight:
                self.layout.setContentsMargins(
                    self.margins['left'], self.margins['top'], self.margins['right'], self.margins['bottom'],
                )
                self.layout.setSpacing((self.margins['wspace'] + self.margins['hspace'])/2.0)
            else:
                nrows = 3  # This isn't actually known, so we have to just guess
                ncols = 3
                spx = (self.margins['right'] - self.margins['left'])/nrows * self.margins['wspace'] * self.width
                spy = (self.margins['top'] - self.margins['bottom'])/ncols * self.margins['hspace'] * self.height
                self.layout.setSpacing((spx + spy)/2.0)
                self.layout.setContentsMargins(
                    self.margins['left']*self.width,
                    (1-self.margins['top'])*self.height,
                    (1-self.margins['right'])*self.width,
                    self.margins['bottom']*self.height,
                )
        return

    def _deleted_axes_protection(self, method=None):
        """Checks whether axes have been deleted by Qt and sets self.axes to None if they have"""
        try:
            tolist(self.axes)
        except RuntimeError:
            m = ' ({})'.format(method) if method is not None else ''
            print('Warning: Qt has deleted the axes; figure had a residual reference to a bad Qt object.{}'.format(m))
            self.axes = None

    def add_subplot(self, nrows, ncols, index, **kwargs):
        """Imitation of matplotlib.figure.Figure.add_subplot"""
        check_unimplemented_keywords(['projection', 'polar'], method='add_subplot', **kwargs)
        row = int(np.floor((index-1)/ncols))
        if row > (nrows-1):
            raise ValueError('index {} would be on row {}, but the last row is {}!'.format(index, row, nrows-1))
        col = (index-1) % ncols
        ax = Axes(nrows=nrows, ncols=ncols, index=index, **kwargs)
        self.layout.addItem(ax, row+1, col)
        self._deleted_axes_protection('add_subplot')
        self.axes = ax if self.axes is None else tolist(self.axes) + [ax]
        self.fig_colspan = max([ncols, self.fig_colspan])
        self.refresh_suptitle()
        return ax

    def colorbar(self, mappable, cax=None, ax=None, **kwargs):
        ax = ax or self.gca()
        if cax is None:
            orientation = kwargs.get('orientation', 'vertical')
            row = int(np.floor((ax.index - 1) / ax.ncols))
            col = (ax.index - 1) % ax.ncols

            cax = Axes(nrows=ax.nrows, ncols=ax.ncols, index=ax.index)
            sub_layout = pg.GraphicsLayout()
            sub_layout.addItem(ax, row=0, col=0)
            sub_layout.addItem(cax, row=int(orientation == 'horizontal'), col=int(orientation != 'horizontal'))
            if orientation == 'horizontal':
                sub_layout.layout.setRowFixedHeight(1, 50)  # https://stackoverflow.com/a/36897295/6605826
            else:
                sub_layout.layout.setColumnFixedWidth(1, 50)  # https://stackoverflow.com/a/36897295/6605826

            # noinspection PyBroadException
            try:
                self.layout.removeItem(ax)
            except Exception:
                pass
            self.layout.addItem(sub_layout, row + 1, col)
        return Colorbar(cax, mappable, **kwargs)

    def suptitle(self, t, **kwargs):
        if len(kwargs.keys()):
            warnings.warn('suptitle keywords are not supported.')
        self.suptitle_text = t
        self.refresh_suptitle()

    def refresh_suptitle(self):
        if self.suptitle_label is not None:
            # noinspection PyBroadException
            try:
                self.layout.removeItem(self.suptitle_label)
            except Exception:  # pyqtgraph raises this type, so we can't be narrower
                pass
        self.suptitle_label = self.layout.addLabel(self.suptitle_text, 0, 0, 1, self.fig_colspan)

    def closeEvent(self, event):
        """
        Intercepts window closing events and updates window tracker
        Not an imitation of matplotlib, but used for interfacing with pyqtgraph behavior
        :param event: window closing event
        """
        printd('window closing')
        tracker.window_closed(self)
        event.accept()
        return

    def gca(self, **kwargs):
        """
        Imitation of matplotlib gca()

        :return: Current axes for this figure, creating them if necessary
        """
        self._deleted_axes_protection('gca')
        if self.axes is not None:
            ax = list(flatten(np.atleast_1d(self.axes)))[-1]
        if self.axes is None:
            ax = self.add_subplot(1, 1, 1, **kwargs)
        return ax

    def close(self):
        self.layout = None
        try:
            super(Figure, self).close()
        except AttributeError:
            # Sometimes this fails the first time, so give 'er the ol' double tap
            super(Figure, self).close()


def check_unimplemented_keywords(unhandled, **kwargs):
    """
    Raises NotImplementedError if keywords include any from a list of unhandled ones (with values other than None)

    :param unhandled: list of strings
        List of unhandled keywords

    :param kwargs:
        Keywords to check
    """
    for uh in unhandled:
        if kwargs.pop(uh, None) is not None:
            raise NotImplementedError('{} keyword in {} is not ready'.format(uh, kwargs.get('method', '?')))
