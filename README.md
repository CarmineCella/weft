# weft

**Companion library for the book *Designing Neural Networks for Music Applications*.**

**W**eights **E**stimated **F**rom **T**raining — a small, dependency-free
neural-network library written from scratch in C++, with every architecture
demonstrated on audio and music applications.

## What this is

A library and a book, written together. The book teaches the major neural
network architectures — multilayer perceptrons, autoencoders, convolutional
networks, attention — through applications that audio and music people care
about: instrument classification, timbre morphing, generative sound texture,
and a capstone that combines matching pursuit with a transformer for
generative audio. The library is the running code: every chapter of the book
points at the same source files; every line in the source files exists
because some chapter of the book explains why.

There are no external dependencies beyond the C++ standard library. No
PyTorch, no NumPy, no BLAS by default. Matrix multiply, backpropagation, the
FFT, convolution, attention — every algorithm is implemented from first
principles and explained in the book.

For why this is written in C++ rather than Python with PyTorch, see
[chapter 1](notes/01_why_cpp.md).

## Principles

- **Header-only and templated.** Each class lives in its own capitalized
  `.h` file (`Matrix.h`, `Layer.h`, `Tensor4D.h`, …), templated on the
  scalar type (`float` or `double`).
- **Standard library only by default.** The default build uses only the
  C++ standard library. An optional `make BLAS=1` opt-in routes matmul
  through Apple Accelerate / OpenBLAS, but every algorithm has a portable
  std-lib path.
- **Readable like the math.** Operator overloading lets a layer's forward
  pass be written literally as `Z = W * X + b`.
- **One namespace.** Everything lives in `namespace weft`.

## Conventions

- A batch of examples is stored **one example per column**: an input batch
  has shape `(features × batch_size)`, a weight matrix has shape
  `(out × in)`, and the forward pass reads `Z = W * X + b`.
- Matrix data is stored in a flat **row-major** buffer for cache
  friendliness.
- 4-D tensors for convolutional layers use **NCHW** layout
  (batch × channels × height × width), matching PyTorch and cuDNN.
  Conversion between 2-D and 4-D is via an explicit `flatten` free
  function — the design choice is discussed in
  [chapter 21](notes/21_from_matrices_to_tensors.md).

## Build and run

```sh
cd build
make             # build all tests and examples
make test        # build and run all tests
make examples    # build the example programs
make clean       # remove build artefacts

make BLAS=1      # opt-in: BLAS-accelerated matmul
                 #   macOS: Apple Accelerate (ships with the OS)
                 #   Linux: OpenBLAS (apt: libopenblas-dev)
```

Tests and examples are both auto-discovered from their folders: drop a
`.cpp` into `tests/` or `examples/` and `make` finds it.

## Project layout

    root/
      src/        *.h        the header-only library
      tests/      *.cpp      one executable per .cpp file
      examples/   *.cpp      one executable per .cpp file
      notes/      *.md       the book's chapters live here
      book/                  LaTeX build of the chapters into one PDF
      build/      Makefile   (you run make from here)
                  tests/     test binaries land here
                  examples/  example binaries land here

## Curriculum

The library is built in the order the book teaches it. Each milestone
produces both classification and generative applications; audio and music
are the binding domain.

| Stage | Material | Chapters |
|---|---|---|
| Why C++ | The language choice, up front and honestly. | 1 |
| Foundations | Matrix algebra, backpropagation, one weight by hand, activations, the Jacobian, cross-entropy, the network class. | 2–8 |
| First classifiers | Iris; optimizers; dropout; MNIST. | 9–12 |
| Audio | The FFT, audio features, an orchestral-instrument classifier. | 13–15 |
| Autoencoders | MNIST autoencoder, variational autoencoder, weight visualization, the audio VAE. | 16–19 |
| Performance | `std::thread`-parallelised matmul and the optional BLAS path. | 20 |
| Convolution | Tensors, Conv2D via im2col, max-pooling, the CIFAR-10 classifier, a convolutional VAE for audio. | 21–23 |
| Differentiable synthesis | A harmonic-plus-noise synthesiser with hand-written gradients, a spectral loss, and a network that plays it: timbre transfer, instrument morphing, notes that were never recorded. | 24 |
| Attention | Multi-head self-attention with a causal mask, positional encoding, layer norm, the transformer block. | 25, 26 |
| Transformer chorales | An autoregressive transformer trained on four-voice Bach chorales, measured against a Markov baseline. | 27 |

Each chapter is one Markdown file in `notes/`, numbered as in the book.

## Status

- **Done:** chapters 1 – 27 and the corresponding code. These are the
  chapters of the book (`book/`, built with `make book`).
- **Not written:** the matching-pursuit capstone and the closing chapter on
  architectures we don't cover (RNNs, GANs, diffusion, reinforcement
  learning).

## Performance

Two performance levers are built in:

- **Threaded matmul** by default — `Matrix::operator*` parallelises across
  rows with `std::thread`. No OpenMP, no architecture-specific intrinsics,
  no external dependency.
- **Optional tuned BLAS** via `make BLAS=1` — the same matmul dispatches to
  `cblas_sgemm` / `cblas_dgemm` through Apple Accelerate (macOS) or
  OpenBLAS (elsewhere). Off by default so the std-lib-only promise holds
  for anyone who doesn't ask for it.

See [chapter 20](notes/20_performance.md) for the full story, including how
profiling led to each choice and why GPU support is deliberately not on
the roadmap.
