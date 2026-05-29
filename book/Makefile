# ============================================================================
#  book/Makefile  --  build the book PDF
#
#  Lives in   book/   and is meant to be run from here, or via the
#  root-level `make book` which delegates here:
#
#      cd book
#      make             # build build/book.pdf
#      make notes       # just regenerate the Part II tex from notes/*.md
#      make clean       # remove generated/ (the per-note tex)
#      make distclean   # remove generated/ AND build/
#
#  Two-stage build:
#    1. preprocess_note.py + pandoc convert ../notes/*.md to generated/*.tex
#       (one chapter per note, with Unicode normalised and heading prefixes
#       stripped).  These are the Part II chapters of the book.
#    2. latexmk runs lualatex on book.tex, which \input{}s
#         chapters/*.tex   (hand-written: preface + Parts I and III)
#         generated/*.tex  (auto-generated: Part II)
#       and produces build/book.pdf.
#
#  Tool requirements:
#    - texlive with lualatex, latexmk, biber, and the packages
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
# Map ../notes/19_convnets.md -> generated/19_convnets.tex
GENERATED := $(patsubst $(NOTES_DIR)/%.md,generated/%.tex,$(NOTES))

# Hand-written chapter sources (Part I + preface; later, Part III + appendices).
# Wrapped in $(wildcard ...) because make does NOT glob bare patterns in
# prerequisite lists -- it would otherwise look for a literal file named
# "chapters/*.tex" and fail.
CHAPTERS  := $(wildcard chapters/*.tex)

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
                --top-level-division=chapter \
                --template pandoc-template.latex

LATEXMK       := latexmk
LATEXMK_FLAGS := -lualatex -interaction=nonstopmode -outdir=$(BUILD_DIR)

#---------------------------------------------------------------------------
.PHONY: all notes clean distclean

all: $(BUILD_DIR)/book.pdf

# Stage 1: regenerate Part II tex from markdown notes.
# The preprocessor handles Unicode math chars and strips heading prefixes
# before pandoc runs.  Pattern rule is per-note so editing one note only
# regenerates that one chapter.
notes: $(GENERATED)

generated/%.tex: $(NOTES_DIR)/%.md pandoc-template.latex preprocess_note.py
	@mkdir -p generated
	python3 preprocess_note.py < $< | $(PANDOC) $(PANDOC_FLAGS) -o $@

# Stage 2: build the PDF.  latexmk handles bibtex/biber and multiple
# passes for cross-references automatically.  We depend on the generated
# tex (so editing a note triggers a rebuild) and on the hand-written
# chapter files (so editing those does too).
$(BUILD_DIR)/book.pdf: book.tex $(GENERATED) $(CHAPTERS)
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(LATEXMK_FLAGS) book.tex
	@echo
	@echo "Built $(BUILD_DIR)/book.pdf"

clean:
	rm -rf generated

distclean: clean
	rm -rf $(BUILD_DIR)
