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

import numpy as np

print(mpl.__version__)
mpl.set_debug(True)

x = np.linspace(0, 10, 101)
y = np.sin(x*np.pi)

plt.plot(x, y)
try:
    scratch['qtapp'] = mpl.app
except NameError:
    print('You must be trying to run the omfit_demo outside of OMFIT, for some reason. amirite?')
else:
    mpl.app.exec_()
