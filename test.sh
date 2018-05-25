#!/bin/bash
python2.7 -m unittest tests/examples
python2.7 -m unittest discover --pattern=*.py -s pgmpl
python2.7 -m unittest discover --pattern=*.py -s tests
