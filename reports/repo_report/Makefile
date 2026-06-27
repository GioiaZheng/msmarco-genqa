# Build the repository report PDF from report.tex + references.bib.
#
#   make pdf    — full PDF build (latexmk runs pdflatex + biber as needed)
#   make html   — standalone HTML build with citations and table of contents
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
PANDOC  ?= pandoc
PYTHON  ?= python
TEX     ?= report.tex
PDF     := $(TEX:.tex=.pdf)
HTML    := $(TEX:.tex=.html)
HTML_TITLE := MS MARCO Generative QA Repository Report
PROJECT_ROOT ?= ../..

.PHONY: pdf html tables clean purge
.DEFAULT_GOAL := pdf

tables:
	cd $(PROJECT_ROOT) && $(PYTHON) scripts/export_report_tables.py

pdf: tables
	$(LATEXMK) -pdf -interaction=nonstopmode $(TEX)

html:
	$(PANDOC) $(TEX) --from=latex --standalone --toc --citeproc \
		--bibliography=references.bib --metadata title="$(HTML_TITLE)" \
		--output=$(HTML)

clean:
	$(LATEXMK) -c $(TEX)
	@rm -f *.bbl *.run.xml

purge: clean
	@rm -f $(PDF)
