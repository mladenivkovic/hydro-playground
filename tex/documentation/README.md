Code Documentation
==================

This directory stores the code documentation. You can build it using the
provided `Makefile`, like so:

```
make
```

That should leave you with the resulting `documentation.pdf` file.

Alternately, you can run the latex compile command by hand:

```
pdflatex -jobname=documentation documentation.tex
```

Another way yet is to import the documentation directory into your LaTeX editor
of choice. The main file you need to import, open and compile is
`documentation.tex`.
