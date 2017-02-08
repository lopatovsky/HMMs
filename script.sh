#!/bin/bash
. env/bin/activate
codeblocks hmms/cthmm.pyx hmms/dthmm.pyx tests/test_cthmm.py tests/common.py tests/test_dthmm.py &
python -m jupyter notebook &
gnome-terminal
gnome-terminal

