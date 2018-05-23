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
import unittest

# Calculation imports
import numpy as np
from matplotlib.cbook import flatten

# Plotting imports
import pyqtgraph as pg
from matplotlib import rcParams

# pgmpl
# noinspection PyUnresolvedReferences
import __init__  # __init__ does setup stuff like making sure a QApp exists
from tracking import tracker
from axes import Axes
from util import printd, tolist
from colorbar import Colorbar


class Figure(pg.PlotWidget):
    """
    Imitates matplotlib.figure.Figure using PyQtGraph
    """
    def __init__(
            self, figsize=None, dpi=None,
            facecolor=None, edgecolor=None, linewidth=0.0,
            frameon=True, subplotpars=None, tight_layout=None, constrained_layout=None,
    ):
        super(Figure, self).__init__()
        self.patch_resize = True  # Controls whether resize events mess with margins or not (this works well now)
        pg.setConfigOption('background', 'w' if facecolor is None else facecolor)
        pg.setConfigOption('foreground', 'k')
        tracker.window_opened(self)
        self.tight = tight_layout or constrained_layout
        if self.tight:
            self.margins = {'left': 10, 'top': 10, 'right': 10, 'bottom': 10, 'hspace': 10, 'wspace': 10}
        else:
            self.margins = None
        self.resizeEvent_original = self.resizeEvent
        self.resizeEvent = self.resize_event
        if dpi is None:
            dpi = rcParams['figure.dpi']
        if figsize is None:
            figsize = rcParams['figure.figsize']
        printd('dpi = {}, figsize = {}'.format(dpi, figsize))
        figsize = np.array(figsize)*dpi
        self.width = figsize[0]
        self.height = figsize[1]
        self.resize(self.width, self.height)
        self.axes = None
        self.layout = None
        self.suptitle_label = None
        self.suptitle_text = ''
        self.fig_colspan = 1
        self.mklay()
        self.clear = self.clearfig  # Just defining the thing as clear doesn't work; needs to be done this way.

        if subplotpars is not None:
            self.set_subplotpars(subplotpars)

        if frameon is not False and linewidth > 0 and edgecolor:
            warnings.warn('WARNING: frame around figure edge is not implemented yet')

    def clearfig(self):
        """Method for clearing the figure. Gets assigned to self.clear"""
        del self.layout
        self.suptitle_text = ''
        self.fig_colspan = 1
        self.mklay()

    def mklay(self):
        """Method for creating layout; used in __init__ and after clear"""
        self.layout = pg.GraphicsLayout()
        self.setCentralItem(self.layout)
        self.show()

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
        if self.layout is None:
            # The layout has already been set to None because the figure is closing. Don't do any margin adjustments.
            return
        fx = self.width
        fy = self.height
        if pars is not None:
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
                spx = (self.margins['right'] - self.margins['left'])/nrows * self.margins['wspace'] * fx
                spy = (self.margins['top'] - self.margins['bottom'])/ncols * self.margins['hspace'] * fy
                self.layout.setSpacing((spx + spy)/2.0)
                self.layout.setContentsMargins(
                    self.margins['left']*fx,
                    (1-self.margins['top'])*fy,
                    (1-self.margins['right'])*fx,
                    self.margins['bottom']*fy,
                )
        return

    def add_subplot(self, nrows, ncols, index, projection=None, polar=None, **kwargs):
        """Imitation of matplotlib.figure.Figure.add_subplot"""
        if projection is not None and projection != 'rectilinear':
            raise NotImplementedError('projection keyword in add_subplot is not ready')
        if polar is not None and polar is not False:
            raise NotImplementedError('polar projection is not ready')
        row = int(np.floor((index-1)/ncols))
        if row > (nrows-1):
            raise ValueError('index {} would be on row {}, but the last row is {}!'.format(index, row, nrows-1))
        col = (index-1) % ncols
        ax = Axes(nrows=nrows, ncols=ncols, index=index, **kwargs)
        self.layout.addItem(ax, row+1, col)
        if self.axes is None:
            self.axes = ax
        else:
            self.axes = tolist(self.axes) + [ax]
        self.fig_colspan = max([ncols, self.fig_colspan])
        self.refresh_suptitle()
        return ax

    def colorbar(self, mappable, cax=None, ax=None, **kwargs):
        if ax is None:
            if self.axes is None:
                ax = self.add_subplot(1, 1, 1)
            else:
                ax = np.atleast_1d(self.axes).flatten()[-1]
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

    def gca(self):
        """
        Imitation of matplotlib gca()
        :return: Current axes for this figure, creating them if necessary
        """
        if self.axes is None:
            ax = self.add_subplot(1, 1, 1)
        else:
            ax = list(flatten(np.atleast_1d(self.axes)))[-1]
        return ax

    def close(self):
        self.layout = None
        try:
            super(Figure, self).close()
        except AttributeError:
            # Sometimes this fails the first time, so give 'er the ol' double tap
            super(Figure, self).close()


class TestPgmplFigure(unittest.TestCase):
    """
    Test from the command line with
    python -m unittest figure
    """

    verbose = False

    def test_figure(self):
        fig1 = Figure()
        assert isinstance(fig1, Figure)
        if self.verbose:
            print('test_figure: fig1 = {}'.format(fig1))
        fig1.close()

    def test_fig_methods(self):
        fig = Figure()
        ax = fig.gca()
        assert isinstance(ax, Axes)
        fig.suptitle('suptitle text in unittest')
        ax2 = fig.gca()
        assert ax2 == ax
        fig.clear()
        fig.close()
        assert fig.clearfig == fig.clear  # Make sure this assignment didn't break.
        if self.verbose:
            print('test_fig_methods: fig = {}, ax = {}'.format(fig, ax))

    def test_fig_plot(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, 1])
        if self.verbose:
            print('test_fig_plot: fig = {}, ax = {}'.format(fig, ax))
        fig.close()

    def test_set_subplotpars(self):
        from matplotlib.figure import SubplotParams
        sp = SubplotParams(left=0.5, right=0.99, bottom=0.01, top=0.5, wspace=0.2, hspace=0.2)
        fig = Figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.plot([0, 1, 0, 1, 0, 1])
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.plot([1, 0, 1])
        fig.set_subplotpars(sp)
        if self.verbose:
            print('test_set_subplotpars: fig = {}, ax = {}, ax2 = {}'.format(fig, ax, ax2))
        fig.close()

    def test_fig_colorbar(self):
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 5)
        a = x[:, np.newaxis] * y[np.newaxis, :]
        img = ax.imshow(a)
        fig.colorbar(img)
        if self.verbose:
            print('test_fig_colorbar: ax = ax')
        fig.close()


if __name__ == '__main__':
    unittest.main()
