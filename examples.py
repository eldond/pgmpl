#!/usr/bin/env python2.7
# # -*- coding: utf-8 -*-

"""
Demonstrations and examples of pyqtmpl usage
"""

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import os

# Calculation imports
import numpy as np

# Plotting imports
import pyqtgraph as pg

# pyqtmpl
import pyqtmpl
import pyqtmpl.pyplot as plt
from pyqtmpl.tracking import tracker

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def demo_plot():
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
    axs[0, 0].legend()

    axs[0, 1].fill_between(x, y1, y2, facecolor=(0, 0, 1, 0.5), edgecolor='k')
    axs[0, 1].fill_between(x, y1+20, y2+25, color='r', alpha=0.4, edgecolor='k', linewidth=3, where=(x > 7) | (x < 4))

    axs[1, 0].plot(x, y1, color='r', label='y1', lw=3)
    axs[1, 0].plot(x, y2, color='k', label='y2')
    axs[1, 0].plot(x, y2+y1, linestyle='--', color='g', label='y2+y1')
    axs[1, 0].plot(x, y3, linestyle='-.', color='b', label='y3')
    axs[1, 0].axvline(np.mean(x), linestyle=':', color='k', label='vline')
    axs[1, 0].axhline(np.mean(y1), linestyle='-', color='k', label='hline')
    axs[1, 0].legend()

    beb = axs[1, 1].errorbar(x, y1, abs(y1) * 0.1, color='b')
    axs[1, 1].errorbar(x, -y2, abs(y2) * 0.1, xerr=0.1, color='r')
    y2line = axs[1, 1].plot(x, y2)
    axs[1, 1].plot(x, y3)
    axs[1, 1].legend([beb, y2line], ['manual label for blue errorbar', 'manual label for y2line'])

    axs[2, 0].plot(x, y1, color='m', marker='o', label='y1 purple circles')
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

    return fig, axs


def short_demo():
    x = np.linspace(0, 2*np.pi, 36)
    fig, axs = plt.subplots(1)
    axs.plot(x, np.cos(x))
    axs.plot(x, np.sin(x))
    fig.clear()
    axs = fig.add_subplot(1, 1, 1)
    axs.plot(x, np.sin(x)+1)
    axs.plot(x, np.cos(x)-1)
    axs.text(0, 0, 'figure cleared then re-used')
    return fig, axs


if __name__ == '__main__':
    print('pyqtmpl examples...')
    pyqtmpl.set_debug(1)
    a = short_demo()
    b = demo_plot()
    # a[0].close()  # This is not needed, but it makes testing faster.
    tracker.status()
    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        print('Starting event loop for pyqtmpl examples...')
        pyqtmpl.app.exec_()
    else:
        print('Done with pyqtmpl examples.')

