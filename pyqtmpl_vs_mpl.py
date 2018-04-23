#!/bin/sh
''':'
exec python2.7 "$0" "$@"

The single quotes on this docstring are a way to trick bash
    https://www.rodneybeede.com/How_to_use_command_line_arguments_with_shebang__usr_bin_env_python.html

To use this script, do ./pyqtmpl_vs_mpl for matplotlib or ./pyqtmpl_vs_mpl pg for pyqtmpl.

This script should work if copied outside of the pyqtmpl folder
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
    use_pyqtmpl = sys.argv[1].lower() in ['pyqtmpl', '1', 'pg', 'p']
else:
    use_pyqtmpl = False

if use_pyqtmpl:
    from PyQt4 import QtGui
    try:
        # This works if you're outside of the pyqtmpl folder, which is what we'd really like to demonstrate
        import pyqtmpl as mpl
        import pyqtmpl.pyplot as plt
    except ImportError:
        # Must be inside of the pyqtmpl area
        import pyplot as plt
else:
    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.pyplot as plt


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

    axs[0, 0].plot(x, y1)
    axs[0, 1].plot(x, y2)
    axs[0, 1].plot(x, y3)

    axs[1, 0].plot(x, y1, color='r')
    axs[1, 0].plot(x, y2, color='k')
    axs[1, 0].plot(x, y2+y1, linestyle='--', color='g')
    axs[1, 0].plot(x, y3, linestyle='-.', color='b')

    axs[1, 1].errorbar(x, y1, abs(y1) * 0.1, color='b')
    axs[1, 1].errorbar(x, -y2, abs(y2) * 0.1, xerr=0.1, color='r')

    axs[2, 0].plot(x, y1, color='m', marker='o')
    axs[2, 1].plot(x, y2, linestyle=' ', color='k', marker='+')
    axs[2, 1].plot(x, y3, linestyle=' ', color='k', marker='x')

    axs[1, 0].axvline(np.mean(x), linestyle=':', color='k')
    axs[1, 0].axhline(np.mean(y1), linestyle='-', color='k')

    if not use_pyqtmpl:
        fig.show()

    return fig, axs


if __name__ == '__main__':
    if os.environ.get('PYQTMPL_DEBUG', None) is None:
        os.environ['PYQTMPL_DEBUG'] = "1"
    if use_pyqtmpl:
        app = QtGui.QApplication(sys.argv)
    else:
        app = None
    b = demo_plot()
    # Start Qt event loop unless running in interactive mode or using pyside.
    if (app is not None) and ((sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION')):
            app.exec_()
    if not use_pyqtmpl:
        plt.ioff()  # https://stackoverflow.com/a/38592888/6605826
        plt.show()
