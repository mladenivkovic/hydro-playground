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
- `python 3` with `numpy` and `matplotlib` for plotting.
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

The entire python module is stored within this repository as a git submodule of its [own
repository](https://github.com/mladenivkovic/mesh_hydro_utils).

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
$ cd python_module $ pip install -e .
```

Alternatively (*albeit very discouraged*), you can add the directory
`./python_module/mesh_hydro_utils` to your `$PYTHONPATH`.




### Getting The Documentation

Check the documentation in `tex/documentation`. You can build it using the provided `Makefile`:

```
$ cd hydro_playground/tex/documentation $ make
```

That should leave you with the resulting `documentation.pdf` file.

Alternately, you can run the latex compile command by hand:

```
$ cd hydro_playground/tex/documentation $ pdflatex -jobname=documentation documentation.tex
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
------------------------------------

TODO. For now, we do the same as in the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro)
code. Look at the top level README.md file there for specifics.




Output
---------------------

TODO. For now, we do the same as in the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro)
code. Look at the top level README.md file there for specifics.




Visualisation
-----------------------------------

Some basic scripts to visualize ICs and outputs are given in the `./python_module/scripts/plotting`
directory. See the `README.md` in the `./python_module/scripts` directory for more details. Note
that for the directory `./python_module/` to contain any files, you first need to initialize the
submodule. See the instructions [above](#Getting-And-Installing-The-Python-Module).





