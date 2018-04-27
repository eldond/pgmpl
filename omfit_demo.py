# -*-Python-*-
# Created by eldond at 2018 Apr 25  11:55

"""
This script demonstrates pgmpl usage within OMFIT.

Load this thing into the OMFIT by executing the following in the OMFIT command box (edit the path):

    OMFIT['mpl_demo'] = OMFITpythonTask('/home/eldond/python_stuff/pgmpl/omfit_demo.py', modifyOriginal=True)
"""

from __future__ import print_function, division
import pgmpl as mpl
import pgmpl.pyplot as plt

print(mpl.__version__)
mpl.set_debug(True)

x = linspace(0, 10, 101)
y = sin(x*pi)

plt.plot(x, y)
scratch['qtapp'] = mpl.app

mpl.app.exec_()
