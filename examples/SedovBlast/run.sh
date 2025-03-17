#!/bin/bash

if [ ! -f sedov.dat ]; then
    python3 ./sedov.py
fi
../../build/hydro --param-file=./sedov.params --ic-file=./sedov.dat
python3 ../../python_module/scripts/plotting/plot_all_results_individually.py output_*out
