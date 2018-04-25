# # -*- coding: utf-8 -*-

# Basic imports
from setuptools import setup

# pyqtmpl imports
import pyqtmpl.__init__

# http://python-packaging.readthedocs.io/en/latest/dependencies.html
# https://packaging.python.org/discussions/install-requires-vs-requirements/
# http://docs.python-guide.org/en/latest/writing/structure/

# Read requirements from requirements.txt
# reqs = open('requirements.txt', 'r')
# rlines = [a for a in reqs.read().split('\n') if len(a)]
with open('requirements.txt') as f:
    required = filter(None, map(lambda x: x.split("#")[0].strip(), f.read().splitlines()))

print required
print('pyqtmpl setup.py...')

# Run setup
setup(
    name='pyqtmpl',
    version=pyqtmpl.__init__.__version__,
    description='Wrapper for calling PyQtGraph with Matplotlib syntax',
    url='https://github.com/eldond/pyqtmpl',
    author=pyqtmpl.__init__.__maintainer__,
    author_email=pyqtmpl.__init__.__email__,
    packages=[
        'pyqtmpl',
    ],
    install_requires=required,
)

print('finished setuptools.setup')
