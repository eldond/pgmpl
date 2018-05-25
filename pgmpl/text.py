#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.text but using PyQtGraph

Classes and methods imitate Matplotlib counterparts as closely as possible, so please see Matplotlib documentation for
more information.
"""

# Basic imports
from __future__ import print_function, division
import unittest
import warnings

# Plotting imports
import pyqtgraph as pg
from translate import color_translator


class Text(pg.TextItem):
    """
    Imitates matplotlib.text.Text using PyQtGraph
    """
    def __init__(
            self, x=0.0, y=0.0, text='',
            color=None,
            verticalalignment='baseline', horizontalalignment='left', multialignment=None,
            fontproperties=None, rotation=None, linespacing=None, rotation_mode=None, usetex=None, wrap=False,
            **kwargs
    ):
        """
        Imitates matplotlib.axes.Axes.text
        :param x: scalar
            Horizontal position in data coordinates
        :param y: scalar
            Vertical position in data coordinates
        :param text: string
            Text to write on the plot
        :param color: string or color spec
            Color of text
        :param verticalalignment: string {top | bottom | center | baseline}
        :param horizontalalignment: string {left | center | right}
        :param multialignment: (not implemented in pgmpl wrapper)
        :param fontproperties: dict (not yet implemented)
        :param rotation: scalar
            Text angle in degrees. 0 is horizontal and upright
        :param linespacing: (not implemented in pgmpl wrapper)
        :param rotation_mode: (not implemented in pgmpl wrapper)
        :param usetex: (not implemented in pgmpl wrapper)
        :param wrap: (not yet implemented in pgmpl wrapper)
        :param kwargs: dict
            Other keywords
        """

        superkw = {
            'text': text,
        }
        t_color = color_translator(**{'color': color})
        if t_color is not None:
            superkw['color'] = t_color
        if rotation is not None:
            superkw['angle'] = rotation
        ha = kwargs.pop('ha', None)
        va = kwargs.pop('va', None)
        horizontalalignment = ha if horizontalalignment is None else horizontalalignment
        verticalalignment = va if verticalalignment is None else verticalalignment
        if horizontalalignment is not None or verticalalignment is not None:
            superkw['anchor'] = (
                {'left': 0, 'center': 0.5, 'right': 1}.get(horizontalalignment, 0),
                {'top': 0, 'bottom': 1, 'center': 0.5, 'baseline': 0.1}.get(verticalalignment, 0.1),
            )
        super(Text, self).__init__(**superkw)
        self.setPos(x, y)

        if multialignment is not None:
            warnings.warn('  pgmpl.text.Text does not support multialignment keyword')
        if rotation_mode is not None:
            warnings.warn('  pgmpl.text.Text does not support rotation_mode keyword')
        if usetex is not None:
            warnings.warn('  pgmpl.text.Text does not support usetex keyword')
        if linespacing is not None:
            warnings.warn('  pgmpl.text.Text does not support linespacing keyword')

        if wrap:
            warnings.warn('  pgmpl.text.Text does not support wrap keyword yet (may be possible later)')
        if fontproperties is not None:
            warnings.warn('  pgmpl.text.Text does not handle font changes yet (to be implemented later)')

        if len(kwargs.keys()):
            warnings.warn('  pgmpl.text.Text got unhandled kwargs: {}'.format(kwargs.keys()))

    def __call__(self):
        return self

