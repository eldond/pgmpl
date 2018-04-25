# # -*- coding: utf-8 -*-

# Basic imports
from setuptools import setup

# pyqtmpl imports
import __init__

# http://python-packaging.readthedocs.io/en/latest/dependencies.html
# https://packaging.python.org/discussions/install-requires-vs-requirements/

# Read requirements from requirements.txt
reqs = open('requirements.txt', 'r')
rlines = [a for a in reqs.read().split('\n') if len(a)]

# Run setup
setup(
    name='pyqtmpl',
    version=__init__.__version__,
    description='Wrapper for calling PyQtGraph with Matplotlib syntax',
    url='https://github.com/eldond/pyqtmpl',
    author=__init__.__maintainer__,
    author_email=__init__.__email__,
    packages=[
        'pyqtmpl',
    ],
    install_requires=rlines,
)
