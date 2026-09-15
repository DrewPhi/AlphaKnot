LATEX ?= pdflatex
BIB ?= bibtex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex refs.bib sections/*.tex
	$(LATEX) main
	-$(BIB) main
	$(LATEX) main
	$(LATEX) main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc
