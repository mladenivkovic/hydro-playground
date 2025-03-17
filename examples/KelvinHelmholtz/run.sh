#!/bin/bash

if [ ! -f kelvin_helmholtz.dat ]; then
    python3 ./kelvin-helmholtz.py
fi
../../build/hydro --param-file=./kh_params.txt --ic-file=./kelvin-helmholtz.dat
python3 ../../python_module/scripts/plotting/plot_all_results_individually.py output_*out
