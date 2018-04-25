#!/usr/bin/env python2.7
# # -*- coding: utf-8 -*-

# Basic imports
from setuptools import setup
import glob
import os

# pyqtmpl imports
from pyqtmpl.info import *


# http://python-packaging.readthedocs.io/en/latest/dependencies.html
# https://packaging.python.org/discussions/install-requires-vs-requirements/
# http://docs.python-guide.org/en/latest/writing/structure/

# Read requirements from requirements.txt (taken from OMFIT setup.py)
with open('requirements.txt') as f:
    required = filter(None, map(lambda x: x.split("#")[0].strip(), f.read().splitlines()))

print required
print('pyqtmpl setup.py...')

# Run setup
setup(
    name='pyqtmpl',
    version=__version__,
    description='Wrapper for calling PyQtGraph with Matplotlib syntax',
    url='https://github.com/eldond/pyqtmpl',
    author=__maintainer__,
    author_email=__email__,
    packages=[
        'pyqtmpl',
    ],
    keywords='plotting plot matplotlib pyqtgraph',
    install_requires=required,
)

print('finished setuptools.setup')
