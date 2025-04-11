TODO ON CUDA BRANCH
================
- put a guard around the linking / compiling with cuda files (should only happen if cuda is really detected)
- Device discovery - how much memory do we have?

- I think we have to configure the grid on the host and then copy over to device!!

- Lost a lot of time to some weird UB stuff on my local machine. 

The way it should work for each class:

Grid:
  - Set up on host as usual
  - Give it a method to move the cell array onto the device
  - We then pass this object BY VALUE into any device code. That way any member attributes we need
    like _nx etc can be copied over (minimal cost) each time we launch a kernel


hydro-playground
================

`hydro-playground` is a simple C++ toy code solving finite volume hydrodynamics in 2D on a uniform
grid. The main goal is to have simple working code as a playground for developments in terms of
acceleration, optimisation, and parallelisation.


It is based on the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro) code, which is a
playground to learn about and experiment with numerical methods to solve hyperbolic conservation
laws, in particular the advection equation and the Euler equations of ideal gases.


Contents
----------------------------

- `./examples`: Some ready-to-go example simulations.
- `./python_module`: A git submodule containing the
  [mesh_hydro_utils](https://github.com/mladenivkovic/mesh_hydro_utils) python module. It contains
  convenience functions to generate initial conditions, plot ICs and outputs, and a Riemann solver.
  Note that you need to install it first for it to work. Instructions are given
  [below](#Getting-And-Installing-The-Python-Module).
- `./src`: contains the actual software.
- `./doc`: Documentation of the code and theory on the equations being solved.
- `./tests`: Unit tests and functional tests.



Getting Started
---------------

### Requirements

- `git` to obtain the code.
- A good old C++ compiler. Code is written in C++20 standard.
- `cmake` 3.21 or above
- (optional) `python 3` with `numpy` and `matplotlib` for plotting outputs and generating initial
  conditions. Additionally with `sphinx` to build the parallelisation documentation.
- (optional) LaTeX to create the TeX files. I hard-coded the `pdflatex` command in the scripts. It
  doesn't require any fancy LaTeX packages.



### Getting The Code

You can get the code from the github repository:

```
git clone https://github.com/mladenivkovic/hydro-playground.git
```

for `https` protocol or

```
git clone git@github.com:mladenivkovic/hydro-playground.git
```

for `ssh`.




### Getting And Installing The Python Module

The entire python module is stored within this repository as a git submodule of its
[own repository](https://github.com/mladenivkovic/mesh_hydro_utils).

Once you've cloned the hydro-playground repository, you'll also need to tell git to grab the
submodules using

```
git submodule init
git submodule update
```

When completed successfully, the directory `./python_module` should now contain some files. We now
need to install this python module.

The easiest way is to navigate into the directory and install it locally using e.g. `pip`:

```
cd python_module
pip install -e .
```

Alternatively (***albeit very discouraged***), you can add the directory
`./python_module/mesh_hydro_utils` to your `$PYTHONPATH`.




### Getting the Documentation

Check the documentation in `doc/tex/documentation`. You can build it using the provided `Makefile`:

```
cd hydro_playground/doc/tex/documentation
make
```

That should leave you with the resulting `documentation.pdf` file.

Alternately, you can run the latex compile command by hand:

```
cd hydro_playground/doc/tex/documentation
pdflatex -jobname=documentation documentation.tex
```

or open the main TeX document, `hydro_playground/doc/tex/documentation/documentation.tex` with your
favourite TeX IDE/Editor.



There is also documentation on parallelisation strategies and paradigms in `hydro_playgournd/doc/rtd`.
You can build `html` or `latex-pdf` outputs using:

```
cd doc/rtd
make html                      # to make html documentation
firefox build/html/index.html  # to view the resulting documentation using firefox
```

for `html` outputs, or

```
cd doc/rtd
make latexpdf                                           # to make latex pdf documentation
okular build/latex/hydro_playground_paralleisation.pdf  # to view the resulting documentation using okular
```



### Building the Project

We build the project using `cmake`:

```
cd hyrdo_playground
mkdir build
cd build
cmake ..
cmake --build .
```

or, if you prefer:

```
cd hyrdo_playground
cmake -B build
cmake --build build
```

That should leave you with an executable file `hydro` in the directory `hydro_playground/build/`.




### Build Options

You can pass build options to `cmake` by giving it a list of command line arguments beginning with
`-D` at build time, e.g.

```
cd hydro_playground
mkdir build
cd build
cmake .. -DOPTION1 -DOPTION2 ...
cmake --build .
```

or, if you prefer:

```
cd hyrdo_playground
cmake -B build -DOPTION1 -DOPTION2 ...
cmake --build build
```


Currently available build options are:

- `-DBUILD_TYPE=` [`Release`, `RelWithDebInfo`, `Debug`] : (Default = `Release`)
  - Build type of the project.
  - `Release`: Enables aggressive compiler optimisation. This is the default mode.
  - `RelWithDebInfo`: Release mode, but with debugging symbols attached. Also activates some light
    debugging checks.
  - `Debug`: Turns compiler optimisation off and enables extensive debugging checks.

- `-DPRECISION=` [`SINGLE`, `DOUBLE`]: (Default=`DOUBLE`)
  - Set precision for floating point variables.
  - `SINLGE`: Single-precision floats.
  - `DOUBLE`: Double-precision floats (Default)

- `-DTERMINAL_COLORS=ON`: Enable coloured output to `stdout` and `stderr` on terminals.

- `-DSOLVER=` [`MUSCL`, `GODUNOV`]: (Default=`MUSCL`)
  - Select hydrodynamics solver.
  - `MUSCL`: MUSCL-Hancock solver (second order accurate)
  - `GODUNOV`: Godunov solver (first order accurate)

- `-DRIEMANN=` [`HLLC`, `EXACT`]: (Default=`HLLC`)
  - Select Riemann solver.
  - `HLLC`: Harten-Lax-van Leer with central wave (approximate solver)
  - `EXACT`: Exact Riemann solver.

- `-DLIMITER=` [`MINMOD`, `VANLEER`]: (Default = `VANLEER`):
  - Select slope limiter for MUSCL-Hancock solver.
  - `MINMOD`: Minmod limiter.
  - `EXACT`: Exact Riemann solver.




### Running an Example

Once you've compiled the code following the steps in the previous sections
([download](#obtaining-the-code) and [install](#building-the-project)), you're ready to run your
first example.

A successful compilation will leave you with an executable `hydro_playground/build/hydro`. To
actually run the code, you need to provide it with two mandatory command line arguments: A
simulation parameter file ([see below for format specification](#parameter-file)) and an initial
conditions file ([see below for format specifications](#initial-conditions)).

You can specify them as follows:

```
./hydro --ic-file <ic_file> --param-file <param_file>
```

or

```
./hydro --ic-file=<ic_file> --param-file=<param_file>
```

where `<ic_file>` is the path to the [initial conditions](#initial-conditions) file you want to use,
and `<param_file>` is the path to the [parameter file](#parameter-file) you want to use.

You may want to look into the `hydro_playground/examples` directory for some ready-to-go examples.





File Format Specifications
----------------------------


### Parameter File

#### Talking Parameters

| name          |  default value    | type  | description                                                                   |
|---------------|-------------------|-------|-------------------------------------------------------------------------------|
| `verbose`     | = 0               | `int` | How talkative the code should be. 0 = quiet, 1 = talky, 2 = no secrets, 3 = debugging        |
|               |                   |       |                                                                               |
| `nstep_log`   | = 0               | `int` | Write log messages only ever `nstep_log` steps. If 0, will write every step.  |
|               |                   |       |                                                                               |



#### Simulation Parameters

| name              |  default value    | type  | description                                                                   |
|-------------------|-------------------|-------|-------------------------------------------------------------------------------|
| `nx`              | = 0               | `int` | Number of cells to use if you're running with a two-state type IC file. Otherwise, it needs to be specified in the initial conditions.  If you're not using a two-state IC, the value will be overwritten by the value given in the IC file.  |
|                   |                   |       |                                                                               |
| `ccfl`            | = 0.9             |`float`| Courant factor; `dt = ccfl * dx / vmax`                                       |
|                   |                   |       |                                                                               |
| `nsteps`          | = 0               | `int` | Up to how many steps to do. If = 0, run until `t >= tmax`                     |
|                   |                   |       |                                                                               |
| `tmax`            | = 0.              |`float`| Up to which time to simulate. If `nsteps` is given, will stop running if `nsteps` steps are reached before `tmax` is.     |
|                   |                   |       |                                                                               |
| `boundary`        | = 0               | `int` | Boundary conditions  0: periodic. 1: reflective. 2: transmissive. This sets the boundary conditions for all walls. |
|                   |                   |       |                                                                               |
| `boxsize`         | = 1.              |`float`| Size of the simulation box in each dimension. Currently unused.               |
|                   |                   |       |                                                                               |
| `replicate`       | = 0               | `int` | When running with [arbitrary-type initial conditions](#arbitrary-type-ics), replicate (= copy-paste) the initial conditions this many times in each dimension. This is not used for the [two-state type ICs](#two-state-type-ics) because there you can simply specify the `nx` parameter as you wish. |
|                   |                   |       |                                                                               |



#### Output Parameters


| name          |  default value    | type    | description                                                                   |
|---------------|-------------------|---------|-------------------------------------------------------------------------------|
| `foutput`     | = 0               | `int`   | Frequency of writing outputs in number of steps. If = 0, will only write initial and final steps.  |
|               |                   |         |                                                                               |
| `dt_out`      | = 0               |`float`  | Frequency of writing outputs in time intervals. Code will always write initial and final steps as well.  |
|               |                   |         |                                                                               |
<!--
| `toutfile`    | None              |`string` | File name containing desired times (in code units) of output. Syntax of the file: One float per line with increasing value.  |
|               |                   |         |                                                                               |
-->
| `basename`    | None              |`string` | Basename for outputs.  If not given, a basename will be generated based on compilation parameters and IC filename.       |
|               |                   |         |                                                                               |
| `write_replications` | false      | `bool`  | If `replicate > 1`, setting this to true will write the entire content of the box, including all replications. |
|               |                   |         |                                                                               |


<!---
#### Source Term Parameters

The source term related options will only take effect if the code has been compiled to add source terms.


| name              |  default value    | type   | description                                                                   |
|-------------------|-------------------|--------|-------------------------------------------------------------------------------|
| `src_const_acc_x` | = 0               | `float`| constant acceleration in x direction for constant source terms                |
|                   |                   |        |                                                                               |
| `src_const_acc_y` | = 0               | `float`| constant acceleration in y direction for constant source terms                |
|                   |                   |        |                                                                               |
| `src_const_acc_r` | = 0               | `float`| constant acceleration in radial direction for radial source terms             |
|                   |                   |        |                                                                               |
-->






### Initial Conditions

In contrast to [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro), `hydro-playground` only
runs 2D examples, and hence only uses 2D initial conditions.

- The program reads two types of IC files.
- In any case, they're expected to be formatted text.
- In both IC file types, lines starting with `//` or `/*` will be recognized as comments and
  skipped. Empty lines are skipped as well.
- Some example python scripts that generate initial conditions are given in
  `./python_module/scripts/IC`. Note that for the directory `./python_module/`
  to contain any files, you first need to initialize the submodule. See the instructions
  [above](#Getting-And-Installing-The-Python-Module).


#### Two-state Type ICs

You can use a Riemann-problem two-state initial condition file as follows:

```
filetype = two-state
rho_L   = <float>
v_L     = <float>
p_L     = <float>
rho_R   = <float>
v_R     = <float>
p_R     = <float>
```

The line
```
filetype = two-state
```

**must** be the first non-comment non-empty line, followed by `rho_L`, `v_L`, `p_L`, `rho_R`, `v_R`,
`p_R`.

The discontinuity between the changes will be in the middle along the `x`-axis. Fluid velocity in `y`
direction will be set to zero, `v_L` and `v_R` will be set as `v_x`.

Note: For "historical" reasons, the velocities can also be specified as `u_L` and `u_R` instead of
`v_L` and `v_R`, respectively.




#### Arbitrary Type ICs

You can provide an individual value for density, velocity, and pressure for each cell. The IC file
format is:

The lines
```
filetype = arbitrary
nx = <int>
ndim = <int>
```

**must** be the first non-comment non-empty lines, in that order. `ndim` **must** be 2, as
`hydro-plaground` only runs in 2D.

The IC format is as follows:


```
filetype = arbitrary
nx = <integer, number of cells in any dimension>
ndim = 2
<density in cell (0, 0)>       <x velocity in cell (0, 0)>      <y velocity in cell (0, 0)>        <pressure in cell (0, 0)>
<density in cell (1, 0)>       <x velocity in cell (1, 0)>      <y velocity in cell (1, 0)>        <pressure in cell (1, 0)>
                                     .
                                     .
                                     .
<density in cell (nx-1, 0)>     <x velocity cell (nx-1, 0)>     <y velocity in cell (nx-1, 0)>     <pressure in cell (nx-1, 0)>
<density in cell (0, 1)>        <x velocity in cell (0, 1)>     <y velocity in cell (0, 1)>        <pressure in cell (0, 1)>
<density in cell (1, 1)>        <x velocity in cell (1, 1)>     <y velocity in cell (1, 1)>        <pressure in cell (1, 1)>
                                     .
                                     .
                                     .
<density in cell (nx-1, 1)>     <x velocity cell (nx-1, 1)>     <y velocity in cell (nx-1, 1)>     <pressure in cell (nx-1, nx-1)>
                                     .
                                     .
                                     .
<density in cell (0, nx-1)>     <x velocity in cell (0, nx-1)>  <y velocity in cell (0, nx-1)>     <pressure in cell (0, nx-1)>
<density in cell (1, nx-1)>     <x velocity in cell (1, nx-1)>  <y velocity in cell (1, nx-1)>     <pressure in cell (1, nx-1)>
                                     .
                                     .
                                     .
<density in cell (nx-1, nx-1)>  <x velocity cell (nx-1, nx-1)>  <y velocity in cell (nx-1, nx-1)>  <pressure in cell (nx-1, nx-1)>
```

`cell (0, 0)` is the lower left corner of the box. First index is x direction, second is y. All
values for density, velocity, and pressure must be floats. You can put comments and empty lines
wherever you feel like it.








### Output Files


If no `basename` is given in the parameter file, the output file name will be generated as follows:

```
output_XXXX.out
```

where `XXXX` is the snapshot/output number.


The output files are written in plain text, and their content should be self-explanatory:


```
# ndim =  2
# nx =    <number of cells used>
# t =     <current time, float>
# nsteps =  <current step of the simulation>
#            x            y          rho          v_x          v_y            p
<x value of cell (0, 0)> <y value of cell (0, 0)> <density in cell (0, 0)> <x velocity in cell (0, 0)> <y velocity in cell (0, 0)> <pressure in cell (0, 0)>
<x value of cell (1, 0)> <y value of cell (1, 0)> <density in cell (1, 0)> <x velocity in cell (1, 0)> <y velocity in cell (1, 0)> <pressure in cell (1, 0)>
                                                 .
                                                 .
                                                 .
<x value of cell (nx-1, 0)> <y value of cell (nx-1, 0)> <density in cell (nx-1, 0)> <x velocity cell (nx-1, 0)> <y velocity in cell (nx-1, 0)> <pressure in cell (nx-1, 0)>
<x value of cell (0, 1)> <y value of cell (0, 1)> <density in cell (0, 1)> <x velocity in cell (0, 1)> <y velocity in cell (0, 1)> <pressure in cell (0, 1)>
<x value of cell (1, 1)> <y value of cell (1, 1)> <density in cell (1, 1)> <x velocity in cell (1, 1)> <y velocity in cell (1, 1)> <pressure in cell (1, 1)>
                                                 .
                                                 .
                                                 .
<x value of cell (nx-1, 1)> <y value of cell (nx-1, 1)> <density in cell (nx-1, 1)> <x velocity cell (nx-1, 1)> <y velocity in cell (nx-1, 1)> <pressure in cell (nx-1, nx-1)>
                                                 .
                                                 .
                                                 .
<x value of cell (0, nx-1)> <y value of cell (0, nx-1)> <density in cell (0, nx-1)> <x velocity in cell (0, nx-1)> <y velocity in cell (0, nx-1)> <pressure in cell (0, nx-1)>
<x value of cell (1, nx-1)> <y value of cell (1, nx-1)> <density in cell (1, nx-1)> <x velocity in cell (1, nx-1)> <y velocity in cell (1, nx-1)> <pressure in cell (1, nx-1)>
                                                 .
                                                 .
                                                 .
<x value of cell (nx-1, nx-1)> <y value of cell (nx-1, nx-1)> <density in cell (nx-1, nx-1)> <x velocity cell (nx-1, nx-1)> <y velocity in cell (nx-1, nx-1)> <pressure in cell (nx-1, nx-1)>
```








Visualisation
--------------------

Some basic scripts to visualize ICs and outputs are given in the `./python_module/scripts/plotting`
directory. See the `README.md` in the `./python_module/scripts` directory for more details. Note
that for the directory `./python_module/` to contain any files, you first need to initialize the
submodule. See the instructions [above](#Getting-And-Installing-The-Python-Module).





