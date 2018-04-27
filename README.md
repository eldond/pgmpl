# PgMpl
A set of wrappers to allow calling of pyqtgraph functions with matplotlib syntax

## Motivation:
Matplotlib is convenient to call and looks nice as a final product, but plots load slowly and features like zoom can be tediously slow as well.
PyQtGraph is much faster, but it's more difficult to get a nice looking final product.
Also, some projects may have a large number of Matplotlib based scripts.
These wrappers are intended to allow a script based on Matplotlib to be easily changed to PyQtGraph for faster development, then changed back to Matplotlib for a publication quality final product.

## Usage:
Import methods like plot from PgMpl instead of from Matplotlib while developing, then switch back to Matplotlib.
PgMpl contains submodules named for the Matplotlib submodules they imitate, like pyplot, figure, and axes.
Hopefully, you can get started by just changing import statements in your script (you may also need to start the `QApplication` event loop).

## Examples and demonstrations:
examples.py is executable and will open a set of demonstration plots if called from the command line.
