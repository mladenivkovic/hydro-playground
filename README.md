hydro-playground
================


`hydro-playground` is a simple toy code solving finite volume hydrodynamics in 1D and 2D in C++. The
main goal is to have simple working code as a playground for developments in terms of acceleration,
optimisation, and parallelisation.


It is based on the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro) code, which is a
playground to learn about and experiment with numerical methods to solve hyperbolic conservation
laws, in particular the advection equation and the Euler equations of ideal gases.


Contents
----------------------------

<!-- - `./IC`: A collection of default initial condition files. -->
- `./src`: contains the actual software.
- `./python_module`: A git submodule containing the
  [mesh_hydro_utils](https://github.com/mladenivkovic/mesh_hydro_utils) python module. It contains
  convenience functions to generate initial conditions, plot ICs and outputs, and a Riemann solver.
  Note that you need to install it first for it to work. Instructions are given
  [below](#Getting-And-Installing-The-Python-Module).
- `./tex`: TeX documentation of the code and theory on the equations being solved.



Getting Started
---------------

TODO.

### Requirements

- A good old C++ compiler. Code is written in C++11 standard.
- `cmake` 3.21 or above
- `python 3` with `numpy` and `matplotlib` for plotting and generating ICs.
- LaTeX to create the TeX files. I hardcoded the `pdflatex` command in the scripts. It doesn't
  require any fancy LaTeX packages.


### Getting The Code

You can get the code from the github repository:

```
$ git clone https://github.com/mladenivkovic/hydro-playground.git
```

or

```
$ git clone git@github.com:mladenivkovic/hydro-playground.git
```



### Getting And Installing The Python Module

The entire python module is stored within this repository as a git submodule of its
[own repository](https://github.com/mladenivkovic/mesh_hydro_utils).

Once you've cloned the mesh-hydro repository, you'll also need to tell git to grab the submodules
using

```
$ git submodule init
$ git submodule update
```

When completed successfully, the directory `./python_module` should now contain some files. We now
need to install this python module.

The easiest way is to navigate into the directory and install it locally using e.g. `pip`:

```
$ cd python_module
$ pip install -e .
```

Alternatively (***albeit very discouraged***), you can add the directory
`./python_module/mesh_hydro_utils` to your `$PYTHONPATH`.




### Getting The Documentation

Check the documentation in `tex/documentation`. You can build it using the provided `Makefile`:

```
$ cd hydro_playground/tex/documentation
$ make
```

That should leave you with the resulting `documentation.pdf` file.

Alternately, you can run the latex compile command by hand:

```
$ cd hydro_playground/tex/documentation
$ pdflatex -jobname=documentation documentation.tex
```



### Building The Project

TODO

Basic steps:

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```




Parameter File
---------------------

TODO. For now, we do the same as in the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro)
code. Look at the top level README.md file there for specifics.





Initial Conditions
---------------------

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


### Two-state ICs

You can use a Riemann-problem two-state initial condition file as follows:

```
filetype = two-state
rho_L   = <float>
u_L     = <float>
p_L     = <float>
rho_R   = <float>
u_R     = <float>
p_R     = <float>
```

The line
```
filetype = two-state
```

**must** be the first non-comment non-empty line. The order of the other parameters is arbitrary,
but they must be named `rho_L`, `u_L`, `p_L`, `rho_R`, `u_R`, `p_R`.

The discontinuity between the changes will be in the middle along the x axis. The coordinates will
be printed to screen.

If the code is supposed to run in 2D, then the split will be along the x axis as well, and just
copied along the y axis. Fluid velocity in y direction will be set to zero, `u_L` and `u_R` will be
set as `u_x`.



### Arbitrary ICs

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
values for density, velocity, and pressure must be floats. You can put comments and empy lines
wherever you feel like it.








Output
--------------------

TODO. For now, we do the same as in the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro)
code. Look at the top level README.md file there for specifics.




Visualisation
--------------------

Some basic scripts to visualize ICs and outputs are given in the `./python_module/scripts/plotting`
directory. See the `README.md` in the `./python_module/scripts` directory for more details. Note
that for the directory `./python_module/` to contain any files, you first need to initialize the
submodule. See the instructions [above](#Getting-And-Installing-The-Python-Module).





