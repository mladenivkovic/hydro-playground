#!/bin/bash

if [ ! -f corner_explosion.dat ]; then
    python3 ./sedov.py
fi
../../build/hydro --param-file=./corner_explosion.txt --ic-file=./corner_explosion.dat
python3 ../../python_module/scripts/plotting/plot_all_results_individually.py output_*out
