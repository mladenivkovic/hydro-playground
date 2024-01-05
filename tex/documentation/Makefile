
TEXFILES = $(wildcard *.tex)

documentation.pdf: ${TEXFILES}
	# pdflatex -jobname=documentation documentation.tex
	# bibtex documentation
	pdflatex -jobname=documentation documentation.tex
	pdflatex -jobname=documentation documentation.tex


.PHONY: clean

clean:
	rm -f documentation.pdf *.aux *.bbl *.blg *.log *.out *.toc  *.synctex.gz 
