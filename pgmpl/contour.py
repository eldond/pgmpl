#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Imitates matplotlib.contour but using PyQtGraph to make the plots.

Classes and methods imitate Matplotlib counterparts as closely as possible, so please see Matplotlib documentation for
more information.
"""


class ContourSet(object):

    def __init__(self, ax, *args, **kwargs):
        self.ax = ax
        pop_default_none = [
            'levels', 'linewidths', 'linestyles', 'alpha', 'origin', 'extent', 'cmap', 'colors', 'norm', 'vmin', 'vmax',
            'antialiased', 'locator',
        ]
        for pdn in pop_default_none:
            self.__setattr__(pdn, kwargs.pop(pdn, None))
        self.filled = kwargs.pop('filled', False)
        self.hatches = kwargs.pop('hatches', [None])
        self.extend = kwargs.pop('extend', 'neither')
        if self.antialiased is None and self.filled:
            self.antialiased = False
        self.nchunk = kwargs.pop('nchunk', 0)
        kwargs = self._process_args(*args, **kwargs)

        return

    def _process_args(self, *args, **kwargs):
        """
        Process *args* and *kwargs*; override in derived classes.

        Must set self.levels, self.zmin and self.zmax, and update axes limits.
        Adapted from matplotlib.contour.ContourSet
        """
        self.levels = args[0]
        self.allsegs = args[1]
        self.allkinds = len(args) > 2 and args[2] or None
        self.zmax = np.max(self.levels)
        self.zmin = np.min(self.levels)
        # self._auto = False
        #
        # # Check lengths of levels and allsegs.
        # if self.filled:
        #     if len(self.allsegs) != len(self.levels) - 1:
        #         raise ValueError('must be one less number of segments as '
        #                          'levels')
        # else:
        #     if len(self.allsegs) != len(self.levels):
        #         raise ValueError('must be same number of segments as levels')
        #
        # # Check length of allkinds.
        # if self.allkinds is not None and len(self.allkinds) != len(self.allsegs):
        #     raise ValueError('allkinds has different length to allsegs')
        #
        # # Determine x,y bounds and update axes data limits.
        # flatseglist = [s for seg in self.allsegs for s in seg]
        # points = np.concatenate(flatseglist, axis=0)
        # self._mins = points.min(axis=0)
        # self._maxs = points.max(axis=0)

        return kwargs


class QuadContourSet(ContourSet):
    """blah"""
#    def __init__(self, **kwargs):
#        super(QuadContourSet, self).__init__(**kwargs)
