#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Information for the pgmpl module.

This would normally go in __init__.py, but importing __init__ in setup.py to get info causes a segfault during package
installation. So, this information is placed here and imported into __init__.py and into setup.py.
"""

# Define module
__version__ = '0.4.0'
__maintainer__ = "David Eldon"
__email__ = "eldond@fusion.gat.com"
__status__ = "Development"

# Set __all__ so these variables can all be imported with *
__all__ = ['__version__', '__maintainer__', '__email__', '__status__']
