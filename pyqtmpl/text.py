#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.text but using PyQtGraph
"""

# Basic imports
from __future__ import print_function, division
import warnings

# Plotting imports
import pyqtgraph as pg
from translate import color_translator


class Text(pg.TextItem):
    """
    Imitates matplotlib.text.Text using PyQtGraph
    """
    def __init__(
            self, x=0, y=0, text='',
            color=None,
            verticalalignment='baseline', horizontalalignment='left', multialignment=None,
            fontproperties=None, rotation=None, linespacing=None, rotation_mode=None, usetex=None, wrap=False,
            **kwargs
    ):
        """
        Imitates matplotlib.axes.Axes.text
        """

        super(Text, self).__init__(
            text=text,
            color=color_translator(**{'color': color}),
            angle=0 if rotation is None else rotation,
            anchor=({'left': 0, 'center': 0.5, 'right': 1}.get(horizontalalignment, 0),
                    {'top': 0, 'bottom': 1, 'center': 0.5, 'baseline': 0.1}.get(verticalalignment, 0.1)),
        )
        self.setPos(x, y)

        if multialignment is not None:
            warnings.warn('  pyqtmpl.text.Text does not support multialignment keyword')
        if rotation_mode is not None:
            warnings.warn('  pyqtmpl.text.Text does not support rotation_mode keyword')
        if usetex is not None:
            warnings.warn('  pyqtmpl.text.Text does not support usetex keyword')
        if linespacing is not None:
            warnings.warn('  pyqtmpl.text.Text does not support linespacing keyword')

        if wrap is not None:
            warnings.warn('  pyqtmpl.text.Text does not support wrap keyword yet (may be possible later)')
        if fontproperties is not None:
            warnings.warn('  pyqtmpl.text.Text does not handle font changes yet (to be implemented later)')

        if len(kwargs.keys()):
            warnings.warn('  pyqtmpl.text.Text got unhandled kwargs: {}'.format(kwargs.keys()))

    def __call__(self):
        return self
