#!/bin/bash
# https://unix.stackexchange.com/a/329576/183163
for script in ./*.py; do
echo
echo Launching test script $script ...
"$script"
done
wait
