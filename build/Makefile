# ============================================================================
#  weft -- Makefile
#
#  Lives in   build/   and is meant to be run from here:
#
#      cd build
#      make            # build all tests and examples
#      make test       # build the tests and run them
#      make examples   # build the example programs
#      make book       # build the LaTeX book in book/build/book.pdf
#      make clean      # remove C++ build artefacts (tests/ and examples/)
#      make book-clean # remove book build artefacts
#      make distclean  # remove everything
#
#  (or from the project root:   make -C build test )
#
#  Project layout this Makefile expects:
#      root/
#        README.md
#        src/        *.h        (the header-only library)
#        tests/      *.cpp      (one executable per .cpp file)
#        examples/   *.cpp      (one executable per .cpp file)
#        notes/      *.md       (canonical chapter sources; Part II of the book)
#        book/       *.tex      (book infrastructure: kaobook + pandoc pipeline)
#                    Makefile   (book-specific build, invoked by `make book` below)
#        build/      Makefile   (you are here)
#                    tests/     (test binaries land here)
#                    examples/  (example binaries land here)
#
#  Tests and examples are both AUTO-DISCOVERED from their folders --
#  adding a new test or a new example is just adding the .cpp file;
#  this Makefile does not need to be touched.
#
#  The `book` target delegates to book/Makefile, which runs pandoc on
#  notes/*.md (producing the Part II chapters) and then lualatex on
#  book/book.tex.  See book/Makefile for tool requirements (pandoc,
#  texlive with lualatex, biber).
# ============================================================================

# ---- locations (relative to this build/ directory) ------------------------
ROOT          := ..
SRC_DIR       := $(ROOT)/src
TEST_DIR      := $(ROOT)/tests
EXAMPLE_DIR   := $(ROOT)/examples
BOOK_DIR      := $(ROOT)/book

# ---- compiler --------------------------------------------------------------
CXX       := g++
# -O3 lets the compiler vectorise the inner loops; -pthread enables the
# std::thread-based parallel matmul in Matrix::operator*.  Both work with
# any standard compiler (Apple Clang, GNU g++, etc.) -- no OpenMP or
# architecture-specific flags required.  For a little extra vectorisation
# you can add -mcpu=native (Apple Clang / Apple Silicon) or -march=native
# (GNU g++ on x86); both are optional and omitted here for portability.
CXXFLAGS  := -std=c++17 -O3 -pthread -Wall -Wextra -I$(SRC_DIR)
LDFLAGS   :=

# Optional tuned-BLAS backend for the matmul:  make BLAS=1
# Off by default, so the standard build stays dependency-free.  When on, it
# defines WEFT_USE_BLAS (activating the BLAS path in Matrix.h) and links a
# BLAS: Apple's Accelerate framework on macOS (ships with the OS, nothing to
# install) or OpenBLAS elsewhere (apt: libopenblas-dev, brew: openblas).
ifdef BLAS
  CXXFLAGS += -DWEFT_USE_BLAS
  ifeq ($(shell uname),Darwin)
    LDFLAGS += -framework Accelerate
  else
    LDFLAGS += -lopenblas
  endif
endif

# ---- the header-only library: tests/examples rebuild if any header changes -
HEADERS   := $(wildcard $(SRC_DIR)/*.h)

# ---- tests: one executable per .cpp in tests/ ------------------------------
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp,tests/%,$(TEST_SRCS))

# ---- examples: one executable per .cpp in examples/ ------------------------
EXAMPLE_SRCS := $(wildcard $(EXAMPLE_DIR)/*.cpp)
EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.cpp,examples/%,$(EXAMPLE_SRCS))

# ---- phony targets ---------------------------------------------------------
.PHONY: all test examples book book-clean clean distclean

all: $(TEST_BINS) $(EXAMPLE_BINS)

# Build all tests AND all examples, then run every test; stop at first failure.
test: $(TEST_BINS) $(EXAMPLE_BINS)
	@echo "--- running tests ---"
	@for t in $(TEST_BINS); do \
		echo "> $$t"; \
		./$$t || exit 1; \
	done
	@echo "--- all tests passed ---"

examples: $(EXAMPLE_BINS)

# ---- book ------------------------------------------------------------------
# Delegates to book/Makefile, which knows about pandoc + lualatex.
# That Makefile does two things:
#   1. Converts ../notes/*.md to LaTeX (via preprocess_note.py + pandoc).
#      This populates book/generated/*.tex (one chapter per note).
#   2. Runs latexmk -lualatex on book/book.tex to produce the PDF.
# Result lands in book/build/book.pdf.
#
# Tool requirements (one-time setup on macOS):
#   - MacTeX (or BasicTeX + texlive-fonts-extra, texlive-publishers,
#     texlive-science, texlive-plain-generic, texlive-bibtex-extra)
#   - pandoc       (brew install pandoc)
#   - Python 3     (for the preprocess_note.py script)
book:
	@echo "--- building book ---"
	$(MAKE) -C $(BOOK_DIR)

book-clean:
	$(MAKE) -C $(BOOK_DIR) distclean

# ---- cleanup ---------------------------------------------------------------
clean:
	rm -rf tests examples

# distclean removes both C++ and book build artefacts.
distclean: clean book-clean

# ---- build rules -----------------------------------------------------------
# `mkdir -p` in the recipe (not a separate prerequisite) keeps `tests` and
# `examples` from being both phony targets AND prerequisites, which would
# loop on themselves.

# Each test .cpp -> its own executable in build/tests/.
tests/%: $(TEST_DIR)/%.cpp $(HEADERS)
	@mkdir -p tests
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Each example .cpp -> its own executable in build/examples/.
examples/%: $(EXAMPLE_DIR)/%.cpp $(HEADERS)
	@mkdir -p examples
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
