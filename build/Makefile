# ============================================================================
#  weft -- Makefile
#
#  Lives in   build/   and is meant to be run from here:
#
#      cd build
#      make            # build all tests and examples
#      make test       # build the tests and run them
#      make examples   # build the example programs
#      make clean      # remove build artefacts
#
#  (or from the project root:   make -C build test )
#
#  Project layout this Makefile expects:
#      root/
#        README.md
#        src/              *.h        (the header-only library)
#        tests/            *.cpp      (one executable per .cpp file)
#        notes/            *.md
#        examples/<name>/  *.cpp      (one executable per example folder)
#        build/            Makefile   (you are here)
# ============================================================================

# ---- locations (relative to this build/ directory) ------------------------
ROOT         := ..
SRC_DIR      := $(ROOT)/src
TEST_DIR     := $(ROOT)/tests
EXAMPLES_DIR := $(ROOT)/examples
BIN_DIR      := bin

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
TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp,$(BIN_DIR)/tests/%,$(TEST_SRCS))

# ---- examples: one executable per folder listed below ----------------------
#  Add new example folders here as we create them (e.g. autoenc, etc.).
ALL_EXAMPLES := \
    backprop_example   \
    iris_classifier    \
    mnist_classifier   \
    extract_features   \
    sound_classifier   \
    mnist_autoencoder  \
    mnist_vae          \
    mnist_weights      \
    train_audio_vae    \
    vae_remap          \
    vae_generate

#  Keep only the ones that actually contain sources yet, so naming a folder
#  before it has any .cpp does not break the build.
EXAMPLES     := $(foreach e,$(ALL_EXAMPLES),$(if $(wildcard $(EXAMPLES_DIR)/$(e)/*.cpp),$(e)))
EXAMPLE_BINS := $(foreach e,$(EXAMPLES),$(BIN_DIR)/examples/$(e))

# ---- phony targets ---------------------------------------------------------
.PHONY: all test examples clean

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

clean:
	rm -rf $(BIN_DIR)

# ---- build rules -----------------------------------------------------------
# Output directories (order-only prerequisites, hence the | ).
$(BIN_DIR)/tests $(BIN_DIR)/examples:
	mkdir -p $@

# Each test .cpp -> its own executable.
$(BIN_DIR)/tests/%: $(TEST_DIR)/%.cpp $(HEADERS) | $(BIN_DIR)/tests
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Each example folder -> one executable built from all .cpp files inside it.
# One rule is generated per example via $(eval).  ($$ escapes for the template.)
define EXAMPLE_template
$(BIN_DIR)/examples/$(1): $$(wildcard $(EXAMPLES_DIR)/$(1)/*.cpp) $$(HEADERS) | $(BIN_DIR)/examples
	$$(CXX) $$(CXXFLAGS) $$(wildcard $(EXAMPLES_DIR)/$(1)/*.cpp) -o $$@ $$(LDFLAGS)
endef
$(foreach e,$(EXAMPLES),$(eval $(call EXAMPLE_template,$(e))))
