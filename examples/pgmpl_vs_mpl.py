#!/bin/sh
''':'
exec python3 "$0" "$@"

The single quotes on this docstring are a way to trick bash
    https://www.rodneybeede.com/How_to_use_command_line_arguments_with_shebang__usr_bin_env_python.html

To use this script, do ./pgmpl_vs_mpl for matplotlib or ./pgmpl_vs_mpl pg for pgmpl.

This script should work if copied outside of the pgmpl folder
'''
# # -*- coding: utf-8 -*-

# Basic imports
from __future__ import print_function, division
import sys
import warnings
import os

# Calculation imports
import numpy as np

print(sys.argv)
if len(sys.argv) > 1:
    use_pgmpl = sys.argv[1].lower() in ['pgmpl', '1', 'pg', 'p']
else:
    use_pgmpl = False

if use_pgmpl:
    from pyqtgraph import QtCore
    import pgmpl as mpl
    import pgmpl.pyplot as plt
    import pgmpl

else:
    import matplotlib as mpl
    if sys.version_info[0] < 3:
        mpl.use('Qt4Agg')
    else:
        mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt


def sample_data():
    x = np.linspace(0, 10, 151)
    y1 = x**2 + 1
    y2 = x*10 - 0.1 * x**3 + 50
    y3 = 85 - y1
    return x, y1, y2, y3


def demo_plot():
    x, y1, y2, y3 = sample_data()
    fig, axs = plt.subplots(3, 2, sharex='col', sharey='row')  # , gridspec_kw={'left': 0.15, 'right': 0.95, 'top': 0.95}, dpi=100)
    fig.suptitle('Drawn using ' + ('pgmpl' if use_pgmpl else 'matplotlib'))
    for ax in axs[-1, :]:
        ax.set_xlabel('x')
    for ax in axs[:, 0]:
        ax.set_ylabel('y')

    axs[0, 0].set_title('plot()')
    axs[0, 0].plot(x, y1)

    axs[0, 1].set_title('plot(), color cycle')
    axs[0, 1].plot(x, y2)
    axs[0, 1].plot(x, y3)

    axs[1, 0].set_title('plot(linestyle=..., color=...)')
    axs[1, 0].plot(x, y1, color='r')
    axs[1, 0].plot(x, y2, color='k')
    axs[1, 0].plot(x, y2+y1, linestyle='--', color='g')
    axs[1, 0].plot(x, y3, linestyle='-.', color='b')

    axs[1, 1].set_title('errorbar() with & without xerr')
    decimate = np.arange(0, len(x)-1, 10)
    axs[1, 1].errorbar(x[decimate], y1[decimate], abs(y1[decimate]) * 0.25, color='b')
    axs[1, 1].errorbar(x[decimate], y1[decimate]-y2[decimate], abs(y2[decimate]) * 0.25, xerr=0.25, color='r')

    axs[2, 0].set_title('plot(marker=...)')
    axs[2, 0].plot(x, y1, color='m', marker='o')
    axs[2, 0].plot(x, y2, linestyle=' ', color='k', marker='+')
    axs[2, 0].plot(x, y3, linestyle=' ', color='k', marker='x')

    axs[2, 1].set_title('axvline and axhline')
    axs[2, 1].axvline(np.mean(x), linestyle=':', color='k')
    axs[2, 1].axhline(np.mean(y1), linestyle='-', color='k')

    if not use_pgmpl:
        fig.show()

    return fig, axs


if __name__ == '__main__':
    if os.environ.get('PGMPL_DEBUG', None) is None:
        os.environ['PGMPL_DEBUG'] = "1"
    if use_pgmpl:
        app = pgmpl.app  # QtGui.QApplication(sys.argv)
    else:
        app = None
    b = demo_plot()
    # Start Qt event loop unless running in interactive mode or using pyside.
    if (app is not None) and ((sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION')):
        app.exec_()
    if not use_pgmpl:
        plt.ioff()  # https://stackoverflow.com/a/38592888/6605826
        plt.show()
