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
x = np.linspace(0, 6.28, 30)
y = x[:]
xx, yy = np.meshgrid(x, y)
z = np.sin(xx) + np.cos(yy)

# Add data
#ax.setXRange(x.min(), x.max())
#ax.setYRange(y.min(), y.max())
c = pg.IsocurveItem(data=z, level=0.5, pen='r', axisOrder='row-major')
# c.setParentItem(ax)  # This doesn't work, of course
ax.addItem(c)

# Finish up
win.show()
sys.exit(app.exec_())
