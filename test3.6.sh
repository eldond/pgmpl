#!/bin/bash
# Script for launching tests in python 3 environment while otherwise working in python 2

# Run the test
export NEWPYTHONPATH=/lib/python/python3.6/site-packages/
export PYTHONPATH=$NEWPYTHONPATH
python3.6 -m unittest discover --pattern=*.py -s tests
export PYTHONPATH=$OLDPYTHONPATH


# More information
# using pip with specific python versions:
#     https://stackoverflow.com/a/33964956/6605826
#     sudo python3.6 -m pip install -r requirements.txt
