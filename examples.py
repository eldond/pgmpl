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

# GUI imports
from PyQt4 import QtGui

# pyqtmpl
import pyplot as plt


def demo_plot():
    x = np.linspace(0, 10, 151)
    y1 = x**2 + 1
    y2 = x*10 - 0.1 * x**3 + 50
    y3 = 85 - y1
    fig, axs = plt.subplots(3, 2, sharex='col', sharey='row', gridspec_kw={'left': 0.25, 'right': 0.95}, dpi=300)
    axs[-1, 0].set_xlabel('x')
    axs[-1, 1].set_xlabel('X')
    axs[0, 0].set_ylabel('y')
    axs[1, 0].set_ylabel('y')
    axs[2, 0].set_ylabel('y')

    axs[0, 0].plot(x, y1, label='y1 label', name='y1 name')
    axs[0, 0].legend()

    axs[0, 1].fill_between(x, y1, y2, facecolor=(0, 0, 1, 0.5), edgecolor='k')
    axs[0, 1].fill_between(x, y1+20, y2+25, color='r', alpha=0.4, edgecolor='k', linewidth=3, where=(x > 7) | (x < 4))

    axs[1, 0].plot(x, y1, color='r', label='y1', lw=3)
    axs[1, 0].plot(x, y2, color='k', label='y2')
    axs[1, 0].plot(x, y2+y1, linestyle='--', color='g', label='y2+y1')
    axs[1, 0].plot(x, y3, linestyle='-.', color='b', label='y3')
    axs[1, 0].legend()

    axs[1, 1].errorbar(x, y1, abs(y1) * 0.1, color='b')
    axs[1, 1].errorbar(x, -y2, abs(y2) * 0.1, xerr=0.1, color='r')
    axs[1, 1].plot(x, y2)
    axs[1, 1].plot(x, y3)

    axs[2, 0].plot(x, y1, color='m', marker='o')
    axs[2, 1].plot(x, y2, linestyle=' ', color='k', marker='+')
    axs[2, 1].plot(x, y3, linestyle=' ', color='k', marker='x')

    axs[1, 0].axvline(np.mean(x), linestyle=':', color='k')
    axs[1, 0].axhline(np.mean(y1), linestyle='-', color='k')

    return fig, axs


def short_demo():
    x = np.linspace(0, 2*np.pi, 36)
    fig, axs = plt.subplots(1)
    axs.plot(x, np.cos(x))
    axs.plot(x, np.sin(x))
    return fig, axs


if __name__ == '__main__':
    if os.environ.get('PYQTMPL_DEBUG', None) is None:
        os.environ['PYQTMPL_DEBUG'] = "1"
    app = QtGui.QApplication(sys.argv)
    a = short_demo()
    b = demo_plot()
    a[0].close()  # This is not needed, but it makes testing faster.
    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            app.exec_()
