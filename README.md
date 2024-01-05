hydro-playground
================


`hydro-playground` is a simple toy code solving finite volume hydrodynamics in
1D and 2D in C++. The main goal is to have simple working code as a playground
for developments in terms of acceleration, optimisation, and parallelisation.


It is based on the [mesh-hydro](https://github.com/mladenivkovic/mesh-hydro)
code, which is a playground to learn about and experiment with numerical methods
to solve hyperbolic conservation laws, in particular the advection equation and
the Euler equations of ideal gases.


Getting Started
---------------

TODO.

For now, check the documentation in `tex/documentation`. You can build it using
the provided `Makefile`:

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


