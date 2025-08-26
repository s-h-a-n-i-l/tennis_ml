# Build with: make        (uses latexmk + biber)
# Clean with: make clean
TEX=main
all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex tex/commands.tex tex/sections/*.tex references.bib
	latexmk -pdf -interaction=nonstopmode -file-line-error -shell-escape $(TEX).tex

biber:
	biber $(TEX)

clean:
	latexmk -C
	@rm -f $(TEX).bbl $(TEX).bcf

.PHONY: all clean biber
