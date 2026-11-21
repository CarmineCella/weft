# ============================================================================
#  book/Makefile  --  build the book PDF
#
#  Lives in   book/   and is meant to be run from here, or via the
#  root-level `make book` which delegates here:
#
#      cd book
#      make             # build build/book.pdf
#      make notes       # just regenerate the chapter tex from notes/*.md
#      make clean       # remove generated/ (the per-note tex)
#      make distclean   # remove generated/ AND build/
#
#  Two-stage build:
#    1. preprocess_note.py + pandoc convert ../notes/*.md to generated/*.tex
#       (one chapter per note, with Unicode normalised and heading prefixes
#       stripped).  These are the chapters of the book.
#    2. latexmk runs pdflatex on book.tex, which \input{}s
#         chapters/*.tex   (hand-written: the introduction)
#         generated/*.tex  (auto-generated: one chapter per note)
#       and produces build/book.pdf.
#
#  Tool requirements:
#    - texlive with pdflatex, latexmk, biber, and the packages
#      texlive-fonts-extra, texlive-publishers, texlive-science,
#      texlive-plain-generic, texlive-bibtex-extra
#      (MacTeX has all of these by default; BasicTeX needs them added.)
#    - pandoc       (brew install pandoc)
#    - python3      (for preprocess_note.py)
#    - fonts: TeX Gyre Pagella + Asana Math + DejaVu Sans + DejaVu Sans Mono
#      (all ship with a standard TeX Live install)
# ============================================================================

NOTES_DIR := ../notes
NOTES     := $(wildcard $(NOTES_DIR)/*.md)
# Map ../notes/22_convnets.md -> generated/22_convnets.tex
GENERATED := $(patsubst $(NOTES_DIR)/%.md,generated/%.tex,$(NOTES))

# Hand-written chapter sources (the introduction).
# Wrapped in $(wildcard ...) because make does NOT glob bare patterns in
# prerequisite lists -- it would otherwise look for a literal file named
# "chapters/*.tex" and fail.
CHAPTERS  := $(wildcard chapters/*.tex)

# The figures, drawn by make_figures.py and kept in the repository.  A
# chapter shows a figure with ![caption](figures/name.png); the
# preprocessor switches to name.pdf when that file exists.
FIGURES   := $(wildcard $(NOTES_DIR)/figures/*)

BUILD_DIR := build

PANDOC    := pandoc
# --listings makes pandoc emit lstlisting environments, which our book.tex
# styles via the listings package.  Recent pandoc versions print a
# deprecation warning suggesting --syntax-highlighting=idiomatic; that
# alternative routes through pandoc's Shaded environment instead and would
# require us to undo the listings setup in book.tex.  The warning is
# harmless; we'll migrate if pandoc actually removes the flag.
PANDOC_FLAGS := --from markdown+tex_math_dollars+raw_tex \
                --to latex \
                --listings \
                --biblatex \
                --top-level-division=chapter \
                --template pandoc-template.latex

LATEXMK       := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -outdir=$(BUILD_DIR)
# Engine: pdflatex.  The preprocess_note.py script handles all Unicode
# in the notes before pandoc, so pdflatex is sufficient (and faster
# than lualatex).  Requires fontmethod=tex in book.tex (set there).

#---------------------------------------------------------------------------
.PHONY: all notes clean distclean

all: $(BUILD_DIR)/book.pdf

# Stage 1: regenerate the chapter tex from markdown notes.
# The preprocessor handles Unicode math chars and strips heading prefixes
# before pandoc runs.  Pattern rule is per-note so editing one note only
# regenerates that one chapter.
notes: $(GENERATED)

generated/%.tex: $(NOTES_DIR)/%.md pandoc-template.latex preprocess_note.py $(FIGURES)
	@mkdir -p generated
	python3 preprocess_note.py $* < $< | $(PANDOC) $(PANDOC_FLAGS) -o $@

# Stage 2: build the PDF.  latexmk handles bibtex/biber and multiple
# passes for cross-references automatically.  We depend on the generated
# tex (so editing a note triggers a rebuild) and on the hand-written
# chapter files (so editing those does too).
#
# The leading `-` on the latexmk line tells make to ignore latexmk's
# exit code.  latexmk will return nonzero whenever pdflatex emits any
# error, including for undefined cross-references -- but the PDF is produced anyway in those
# cases.  We check for the actual PDF after the run; if it's there, the
# build succeeded; if not, fail.
$(BUILD_DIR)/book.pdf: book.tex $(GENERATED) $(CHAPTERS) $(FIGURES)
	@mkdir -p $(BUILD_DIR)
	@# An editor building book.tex in place leaves book.bbl / book.bcf here.
	@# LaTeX finds those before the ones in $(BUILD_DIR), and latexmk then
	@# runs bibtex instead of biber and stops after one pass.
	@rm -f book.bbl book.bcf
	-$(LATEXMK) $(LATEXMK_FLAGS) book.tex
	@if [ -f $(BUILD_DIR)/book.pdf ]; then \
	    echo ""; \
	    echo "Built $(BUILD_DIR)/book.pdf"; \
	    echo "(Any 'Undefined refs' messages above are warnings, not errors.)"; \
	else \
	    echo "BUILD FAILED: no PDF produced.  See $(BUILD_DIR)/book.log."; \
	    exit 1; \
	fi

clean:
	rm -rf generated

distclean: clean
	rm -rf $(BUILD_DIR)
