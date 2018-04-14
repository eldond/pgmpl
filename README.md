# PyQtMpl
A set of wrappers to allow calling of pyqtgraph functions with matplotlib syntax

## Motivation:
Matplotlib is convenient to call and looks nice as a final product, but plots load slowly and features like zoom can be tediously slow as well.
PyQtGraph is much faster, but it's more difficult to get a nice looking final product.
Also, some projects may have a large number of Matplotlib based scripts.
These wrappers are intended to allow a script based on Matplotlib to be easily changed to PyQtGraph for faster development, then changed back to MatPlotLib for a publication quality final product.

## Usage:
Import methods like plot from PyQtMpl instead of from Matplotlib while developing, then switch back to Matplotlib.
