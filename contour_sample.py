from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys

# Setup
app = QtGui.QApplication([])
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

win = pg.PlotWidget()
layout = pg.GraphicsLayout()
win.setCentralItem(layout)
ax = pg.PlotItem()
layout.addItem(ax)

# Generate data
x = np.linspace(10, 16.28, 30)
y = x[:]
xx, yy = np.meshgrid(x, y)
z = np.sin(xx) + np.cos(yy)

# Add data
ax.setXRange(x.min(), x.max())
ax.setYRange(y.min(), y.max())

c = pg.IsocurveItem(data=z, level=0.5, pen='r', axisOrder='row-major')
img = pg.ImageItem(z, axisOrder='row-major')
img.translate(x.min(), y.min())
img.scale((x.max() - x.min()) / img.width(), (y.max() - y.min()) / img.height())
ax.addItem(img)

# c.setParentItem(img)
# https://stackoverflow.com/a/51109935/6605826
c.translate(x.min(), y.min())
c.scale((x.max() - x.min()) / np.shape(z)[0], (y.max() - y.min()) / np.shape(z)[1])

ax.addItem(c)

# Finish up
win.show()
sys.exit(app.exec_())
