#!/usr/bin/env bash
# Rebuild the paper: figures -> architecture diagram -> main.pdf (run from paper/)
set -e
cd "$(dirname "$0")/.."
python scripts/extract_tiles.py
python scripts/make_figures.py
python scripts/arch_panels.py
(cd figures && pdflatex -interaction=nonstopmode -halt-on-error fig_arch.tex > /dev/null)
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
grep -E "Output written|undefined" main.log || true
