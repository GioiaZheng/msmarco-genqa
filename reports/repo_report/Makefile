# Build the repository report PDF from report.tex + references.bib.
#
#   make pdf    — full build (latexmk runs pdflatex + biber as needed)
#   make clean  — remove all intermediate artefacts (keeps report.pdf)
#   make purge  — remove intermediates AND report.pdf
#
# Requires: pdflatex + biber + biblatex.  TinyTeX (user-mode, no sudo)
# installs all of these via `curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh | sh`
# followed by:
#   tlmgr install biblatex biber booktabs multirow tabularx longtable \
#                 makecell microtype geometry fancyhdr float caption \
#                 subcaption listings xcolor hyperref cleveref grfext \
#                 epstopdf-pkg parskip titlesec seqsplit hyphenat \
#                 collectbox subfig oberdiek zref

LATEXMK ?= latexmk
PYTHON  ?= python
TEX     ?= report.tex
PDF     := $(TEX:.tex=.pdf)
PROJECT_ROOT ?= ../..

.PHONY: pdf tables clean purge
.DEFAULT_GOAL := pdf

tables:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/export_report_tables.py

pdf: tables
	$(LATEXMK) -pdf -interaction=nonstopmode $(TEX)

clean:
	$(LATEXMK) -c $(TEX)
	@rm -f *.bbl *.run.xml

purge: clean
	@rm -f $(PDF)
