#!/bin/bash
codeblocks cthmm.pyx dthmm.pyx tests/test_cthmm.py tests/common.py tests/test_dthmm.py &
. env/bin/activate
gnome-terminal
gnome-terminal
python -m jupyter notebook

