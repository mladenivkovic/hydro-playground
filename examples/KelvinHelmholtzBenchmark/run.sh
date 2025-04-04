#!/bin/bash

if [ ! -f kelvin_helmholtz.dat ]; then
    python3 ./kelvin_helmholtz.py
fi
../../build/hydro --param-file=./kh.params --ic-file=./kelvin_helmholtz.dat
python3 ../../python_module/scripts/plotting/plot_all_results_individually.py output_*out
