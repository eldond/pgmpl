#!/usr/bin/env python2.7
# # -*- coding: utf-8 -*-

"""
Demonstrations and examples of pgmpl usage
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import os
import unittest

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg

# pgmpl
import pgmpl
import pgmpl.util
import pgmpl.pyplot as plt
from pgmpl.tracking import tracker

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def demo_plot():
    """
    Big, multi-panel demo plot that uses many different methods
    """
    x = np.linspace(0, 10, 151)
    y1 = x**2 + 1
    y2 = x*10 - 0.1 * x**3 + 50
    y3 = 85 - y1
    fig, axs = plt.subplots(3, 2, sharex='col', sharey='row', gridspec_kw={'left': 0.25, 'right': 0.95}, dpi=150)
    axs[-1, 0].set_xlabel('x')
    axs[-1, 1].set_xlabel('X')
    axs[0, 0].set_ylabel('y')
    axs[1, 0].set_ylabel('y')
    axs[2, 0].set_ylabel('y')

    axs[0, 0].plot(x, y1, label='y1 label', name='y1 name', lw=2.5)
    axs[0, 0].text(5, 20, 'red text @ (5, 20)', color='r')
    axs[0, 0].text(1, 30, '45 deg', color='b', rotation=45)
    axs[0, 0].set_title('axs00 title')
    axs[0, 0].legend()
    axs[0, 0].set_aspect('equal')

    axs[0, 1].fill_between(x, y1, y2, facecolor=(0, 0, 1, 0.5), edgecolor='k')
    axs[0, 1].fill_between(x, y1+20, y2+25, color='r', alpha=0.4, edgecolor='k', linewidth=3, where=(x > 7) | (x < 4))

    axs[1, 0].plot(x, y1, color='r', label='y1', lw=3)
    axs[1, 0].plot(x, y2, color='k', label='y2')
    axs[1, 0].plot(x, y2+y1, linestyle='--', color='g', label='y2+y1')
    axs[1, 0].plot(x, y3, linestyle='-.', color='b', label='y3')
    axs[1, 0].axvline(np.mean(x), linestyle=':', color='k', label='vline')
    axs[1, 0].axhline(np.mean(y1), linestyle='-', color='k', label='hline')
    axs[1, 0].legend()

    beb = axs[1, 1].errorbar(x, y1, abs(y1) * 0.1, color='b', errorevery=10)
    axs[1, 1].errorbar(x, -y2, abs(y2) * 0.1, xerr=0.1, color='r')
    y2line = axs[1, 1].plot(x, y2)
    axs[1, 1].plot(x, y3)
    axs[1, 1].legend([beb, y2line], ['manual label for blue errorbar', 'manual label for y2line'])
    axs[1, 1].errorbar(-x[0:3], y1[0:3], y1[0:3] * 0.25, color='m', uplims=True, marker='o', ecolor='r')
    axs[1, 1].errorbar(-x[3:6], y1[3:6], y1[3:6] * 0.25, color='k', lolims=True, capsize=5)
    axs[1, 1].errorbar(-x[6:9], y1[6:9], y1[6:9] * 0.25, color='y', lolims=True, uplims=True, capsize=10, capthick=2)
    axs[1, 1].errorbar(-x[20:25], y1[20:25], y1[20:25] * 0.1, x[20:25] * 0.05, color='r', xlolims=True)
    axs[1, 1].errorbar(-x[30:35], y1[30:35], y1[30:35] * 0.1, x[30:35] * 0.05, color='b', xuplims=True)
    axs[1, 1].errorbar(-x[40:45], y1[40:45], y1[40:45] * 0.1, x[40:45] * 0.05, color='g', xuplims=True, xlolims=True)

    axs[2, 0].plot(x, y1, color='m', marker='o', label='y1 purple circles')
    axs[2, 0].scatter([0, 2, 4, 6, 8, 10], [80, 40, 90, 10, 20, 5], c=['r', 'b', 'g', 'k', 'm', 'y'], linewidths=1)
    axs[2, 0].scatter(x, x*0+60, c=x, marker='v', s=4, edgecolors=' ')
    axs[2, 0].scatter(x, x*0+50, c=x, marker='s', s=5, cmap='plasma', linewidths=0)
    axs[2, 0].scatter([0, 2, 4, 6, 8, 10], [50, 90, 80, 0, 50, 90], c=[90, 0, 50, 20, 75, 66],
                      marker='^', linewidths=2, edgecolors='r')
    axs[2, 0].scatter(x, x*0-10, c=x, cmap='jet', marker=None,
                      verts=[(0, 0), (0.5, 0.5), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, -0.5)])

    leg = axs[2, 0].legend()
    leg.draggable()

    axs[2, 1].plot(x, y2, linestyle=' ', color='k', marker='+', label='y2 blk +')
    axs[2, 1].plot(x, y3, linestyle=' ', color='k', marker='x', label='y3 blk x')
    axs[2, 1].errorbar(x, -y2, abs(y2) * 0.1, xerr=0.1, color='r', label='-y2 red err bar')
    axs[2, 1].legend()

    axs[2, 1].clear()
    axs[2, 1].text(0, 0, 'these axes were cleared', color='k')
    axs[2, 1].plot([0, 1], [1, 0], label='re-added after clear')
    axs[2, 1].legend()
    axs[2, 1].set_aspect(10)

    fig.suptitle('super title of the figure')
    fig.suptitle('suptitle 2: the replacement suptitle (the old one should be gone)')

    return fig, axs


def short_demo():
    """
    Minimal demo with just one panel
    """
    x = np.linspace(0, 2*np.pi, 36)
    fig, axs = plt.subplots(1)
    axs.plot(x, np.cos(x))
    axs.plot(x, np.sin(x))
    fig.clear()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x, np.sin(x)+1)
    axs.plot(x, np.cos(x)-1)
    axs.text(0, 0, 'figure cleared then re-used')
    fig.suptitle('suptitle')
    return fig, axs


def log_demo():
    """
    Test log scaling of axes
    """
    x = np.linspace(0, 10, 101)
    y = np.exp(x)
    fig, ax = plt.subplots(1)
    ax.plot(x, y)
    ax.set_yscale('log')
    ax.set_title('log demo')
    ax.set_ylabel('should be log scaled')
    ax.set_xlabel('should be linearly scaled')
    return fig, ax


def twod_demo():
    """
    Test 2D plotting methods like imshow
    """
    nx = 40
    ny = 20
    a = np.zeros((ny, nx, 3))
    a[0, 0, :] = 0.9
    a[4, 4, :] = 1
    a[3, 3, :] = 0.5
    a[10, 1, :] = 0.7
    a[2, 11, 0] = 0.65
    a[15, 5, :] = 0.1
    a[5, 15, :] = 0.2
    a[18:20, 0:2, 0] = 0.55
    a[18, 1, 0:2] = 0.9
    for i in range(3):
        a[:, int(np.floor(nx*5.0/6))+i, i] = np.linspace(0, 1, ny)
    fig, axs = plt.subplots(2, 2)
    img = axs[0, 0].imshow(a[:, :, 0], origin='lower')
    axs[0, 1].imshow(a[:, :, 0], origin='upper')
    img2 = axs[1, 0].imshow(a[:, :, 0], extent=[10, 12, 20, 22], origin='lower')
    axs[1, 1].imshow(a, extent=[10, 12, 20, 22], origin='lower')
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 1].set_aspect('auto', adjustable='box')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 1].set_aspect('auto', adjustable='box')
    fig.colorbar(img, ax=axs[0, 0], label='colorbar label~~~')
    fig.colorbar(img2, ax=axs[1, 0], label='horz cb label :)', orientation='horizontal')

    return fig, axs


def contour_demo():
    """
    Test contour-based plot methods
    """
    x = np.linspace(0, 1.8, 30)
    y = np.linspace(1, 3.1, 25)
    z = (x[:, np.newaxis] - 0.94)**2 + (y[np.newaxis, :] - 2.2)**2 + 1.145
    levels = [1.2, 1.5, 2, 2.5, 2.95]
    nlvl = len(levels) * 4

    fig, axs = plt.subplots(2, 2)
    return fig, axs


def open_examples(close_after=False, start_event=True):
    print('pgmpl examples...')
    pgmpl.util.set_debug(0)

    a = short_demo()
    b = demo_plot()
    c = log_demo()
    d = twod_demo()
    pgmpl.util.printd('  Example plots: a = {}, b = {}, c = {}, d = {}'.format(a, b, c, d))
    tracker.status()

    if close_after:
        a[0].close()
        b[0].close()
        c[0].close()
        d[0].close()
        tracker.status()
    elif start_event:
        # Start Qt event loop unless running in interactive mode or using pyside.
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            print('Starting event loop for pgmpl examples...')
            pgmpl.app.exec_()
        else:
            print('Done with pgmpl examples.')


class TestPgmplExamples(unittest.TestCase):

    verbose = int(os.environ.get('PGMPL_TEST_VERBOSE', '0'))

    def test_demo_plot(self):
        """
        Just call them all
        """
        open_examples(close_after=True)
        if self.verbose:
            print('  Tested examples.py')


if __name__ == '__main__':
    open_examples(start_event=True)
