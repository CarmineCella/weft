# weft

[![CI](https://github.com/CarmineCella/weft/actions/workflows/ci.yml/badge.svg)](https://github.com/CarmineCella/weft/actions/workflows/ci.yml)

Companion library for the book *Designing Musical Networks: neural
networks for sound and music, built from scratch in C++*.

weft (**W**eights **E**stimated **F**rom **T**raining) is a small
neural-network library written from scratch in C++. It has no
dependencies beyond the C++ standard library. Every architecture in it
is used on an audio or music task.

## What this is

A library and a book, written together. The book covers multilayer
perceptrons, autoencoders, convolutional networks and attention. Each
is applied to a task from audio or music: instrument classification,
timbre morphing, a network that plays a synthesiser, a transformer that
writes four-voice chorales, and a system that combines matching pursuit
with a transformer to generate sound.

Each chapter of the book explains part of the library, and each part of
the library is explained by a chapter.

Begin with the [listening prologue](notes/00_listening.md) and its
[offline audio player](notes/audio/index.html). The refreshed MNIST,
dense audio VAE, and differentiable-synthesis experiments have a
[fixed reproduction protocol](book/experiments/README.md), recorded
commands, file hashes, logs, and results.

The [concluding chapter](notes/30_conclusions.md) brings together the
representation choices, experimental limits, and possible next projects.

The matrix product, backpropagation, the FFT, convolution and attention
are implemented in the library and explained in the book. PyTorch,
NumPy and BLAS are not used by the default build.

[Chapter 1](notes/01_why_cpp.md) explains why the library is in C++ and
not in Python.

## Principles

- **Header-only and templated.** Each class is in its own `.h` file
  (`Matrix.h`, `Layer.h`, `Tensor4D.h`, ...), templated on the scalar
  type (`float` or `double`).
- **Standard library only by default.** `make BLAS=1` optionally sends
  the matrix product to Apple Accelerate or OpenBLAS. Every algorithm
  also has a portable version.
- **Code that reads like the mathematics.** Operator overloading lets a
  layer's forward pass be written as `Z = W * X + b`.
- **One namespace.** Everything is in `namespace weft`.

## Conventions

- A batch stores one example per column. An input batch has shape
  `(features × batch_size)`, a weight matrix has shape `(out × in)`, and
  the forward pass is `Z = W * X + b`.
- Matrix data is stored in one flat buffer in row-major order.
- 4-D tensors for convolutional layers use the NCHW layout
  (batch × channels × height × width). The free function `flatten`
  converts a 4-D tensor to a matrix;
  [chapter 21](notes/21_from_matrices_to_tensors.md) discusses this
  choice.

## Build and run

```sh
cd build
make             # build all tests and examples
make test        # build and run all tests
make headers     # compile each public header independently
make examples    # build the example programs
make book        # build the book (needs LaTeX and pandoc)
make clean       # remove build artefacts

make BLAS=1      # optional: BLAS matrix product
                 #   macOS: Apple Accelerate (part of the system)
                 #   Linux: OpenBLAS (apt: libopenblas-dev)
```

Tests and examples are found automatically: a `.cpp` file placed in
`tests/` or `examples/` is built by `make`.

## Classification and reproducibility

Classification examples return **logits** from their final Dense layer
and train with `SoftmaxCrossEntropy`. It evaluates log-sum-exp directly
and returns the logits gradient, including batch averaging. Do not add
a Softmax layer before this loss. Its `probabilities()` accessor provides
the corresponding normalized predictions after `forward`. The standalone
Softmax and probability-space CrossEntropy remain available for study.

Call `weft::set_seed(42)` before constructing a model to reproduce its
initialization and dropout streams. Explicit per-layer seeds remain
available. Reproducibility is within a given toolchain/backend; standard
random distributions and floating-point reductions can differ between
platforms.

The classifier examples monitor validation data and reserve test data
for the final evaluation. MNIST saves the checkpoint with the lowest
validation loss:

```sh
./examples/mnist_classifier ../data/mnist 42 mnist.wnet
```

## Saved models and audio features

New Network, ConvNetwork, and TokenTransformer checkpoints validate
their version, scalar width, and architecture/configuration. Parameter
files contain weights, not optimizer moments or a resumable RNG state.
Existing version-one dense/conv files and headerless transformer files
can still be loaded through their legacy paths with the same architecture.

Feature-cache version 3 records sample rate, FFT/hop settings, window,
and original frame offsets. Temporal patches split at file boundaries
and at gaps left by silence removal. Version-two caches remain readable
for classification; re-run extraction for the audio VAE examples.
Those examples share `AudioModelConfig` and check the preprocessing
metadata stored with each model. They require 44.1 kHz input.
Old audio weights need verified metadata before reuse; see
[the release notes](CHANGELOG.md).

## Continuous integration

GitHub Actions builds all examples and runs the test programs on Linux
(GCC and Clang) and macOS (Clang), including OpenBLAS/Accelerate builds.
It also checks independent header compilation, runs synthetic-data
example pipelines, validates the Markdown chapters, and compiles the
book PDF as a downloadable workflow artifact.
The book job also verifies the recorded experiment evidence and audio
hashes, and packages a ZIP containing the PDF and its offline listening
files. Dataset training runs are reproduced with the protocol above;
they are not rerun by CI.

The end-to-end smoke checks need Python 3 and no downloaded datasets:

```sh
make -C build test headers
python3 tests/smoke_examples.py
python3 book/check_sources.py
```

## Data

The data sets are not in the repository. Each has a script in `data/`:

| Script | Data | Used in chapters |
| --- | --- | --- |
| `download_mnist.sh` | MNIST handwritten digits | 12, 16–18 |
| `download_TinySOL_2020.sh` | TinySOL instrument samples | 15, 19, 23, 24 |
| `download_cifar10.sh` | CIFAR-10 images | 22 |
| `make_chorales.py` | Bach chorales as MIDI, from the music21 corpus | 27 |

`make_chorales.py` needs Python (`pip install music21`). The library,
the tests and the examples do not. Python is used in one other place:
`book/make_figures.py` draws the figures of the book from the files the
examples write. The figures are in `notes/figures/`, so the book builds
without running that script. Three short recordings in `data/` are used as
test sounds in chapters 19, 23, 28 and 29.

## Project layout

    src/        *.h        the library
    tests/      *.cpp      one test program per file
    examples/   *.cpp      one example program per file
    notes/      *.md       the chapters of the book, one file each
                figures/   the figures of the book
    book/                  builds the chapters into one PDF
    data/                  download scripts and test sounds
    build/      Makefile   run make here
                tests/     test programs are built here
                examples/  example programs are built here

## Contents

The library is built in the order of the chapters. Each chapter is one
Markdown file in `notes/`, with the same number as in the book.

| Part | Material | Chapters |
| --- | --- | --- |
| Foundations | Why C++; matrices; backpropagation; one weight by hand; activations; the Jacobian; cross-entropy; the network class. | 1–8 |
| Training and first classifiers | Iris; optimisers; dropout; MNIST. | 9–12 |
| Sound and autoencoders | The FFT; audio features; an instrument classifier; autoencoder; variational autoencoder; weight visualisation; the audio VAE. | 13–19 |
| Convolution | A threaded matrix product and the optional BLAS path; tensors; Conv2D by im2col, max-pooling and a CIFAR-10 classifier; a convolutional VAE for audio. | 20–23 |
| Synthesis and sequence models | A network that controls a harmonic-plus-noise synthesiser; attention; the transformer block; a transformer trained on Bach chorales; matching pursuit; a transformer trained on matching-pursuit atoms. | 24–29 |
| Conclusions | Choosing representations, interpreting results, and extending the experiments. | 30 |

Recurrent networks, GANs, diffusion models and reinforcement learning
are not covered.

## Performance

- **Threaded matrix product.** `Matrix::operator*` splits the rows of
  the result across threads with `std::thread`. It uses no OpenMP and
  no processor-specific instructions.
- **Optional BLAS.** With `make BLAS=1` the same product calls
  `cblas_sgemm` or `cblas_dgemm` from Apple Accelerate (macOS) or
  OpenBLAS (elsewhere). It is off by default so that the default build
  needs nothing beyond the standard library.

[Chapter 20](notes/20_performance.md) describes the profiling behind
each choice and why there is no GPU support.
