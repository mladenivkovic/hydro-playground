
TEXFILES = $(wildcard *.tex)

theory.pdf: ${TEXFILES}
	# pdflatex -jobname=documentation documentation.tex
	# bibtex documentation
	pdflatex -jobname=theory theory.tex
	pdflatex -jobname=theory theory.tex


.PHONY: clean

clean:
	rm -f theory.pdf *.aux *.bbl *.blg *.log *.out *.toc  *.synctex.gz 
