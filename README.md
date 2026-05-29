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
principles and explained in the notes.

For why this is written in C++ rather than Python with PyTorch, see
[note 0b](notes/0b_why_cpp.md).

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
  [note 18](notes/18_from_matrices_to_tensors.md).

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
      build/      Makefile   (you run make from here)
                  tests/     test binaries land here
                  examples/  example binaries land here

## Curriculum

The library is built in the order the book teaches it. Each milestone
produces both classification and generative applications; audio and music
are the binding domain.

| Stage | Material | Notes |
|---|---|---|
| Why C++ | The language choice, up front and honestly. | 0b |
| Foundations | Matrix algebra, backpropagation, layers and activations, losses, the network class, optimizers, dropout. | 01–08 |
| MLP applications | Iris classifier; FFT and audio features; MNIST classifier; orchestral-instrument sound classifier. | 06, 09, 10, 11, 12 |
| Autoencoders | MNIST autoencoder, weight visualization, dense variational autoencoder. | 13, 14, 15 |
| Audio VAE | Spectrogram VAE for timbre interpolation, generation, and remap on real instrument samples. | 16 |
| Performance | `std::thread`-parallelised matmul and the optional BLAS path. | 17 |
| Convolutional networks | Tensors, Conv2D via im2col, max-pooling, the MNIST conv classifier. | 18, 19 |
| Convolutional VAE | Conv-based VAE for audio spectrograms; sharper morphs and interpolations than the dense one. | 20 |
| Attention | Multi-head self-attention, positional encoding, layer norm, the transformer block. | 21 |
| Capstone | Matching pursuit as an interpretable codec, plus a transformer learning the temporal structure of atom sequences — a generative audio system. | 22 |
| Architectures we don't cover | RNNs, GANs, diffusion, reinforcement learning — what they are, why we skipped them, where to read more. | 23 |

## Status

- **Done:** notes 01 – 18 and the corresponding code (foundations, MLP
  applications, dense AE/VAE, audio VAE, performance, tensors).
- **In progress:** note 19 (Conv2D, MaxPool2D, MNIST conv classifier).
- **Planned:** notes 20 – 23 (ConvVAE, attention, MP + attention capstone,
  architectures we don't cover).

## Performance

Two performance levers are built in:

- **Threaded matmul** by default — `Matrix::operator*` parallelises across
  rows with `std::thread`. No OpenMP, no architecture-specific intrinsics,
  no external dependency.
- **Optional tuned BLAS** via `make BLAS=1` — the same matmul dispatches to
  `cblas_sgemm` / `cblas_dgemm` through Apple Accelerate (macOS) or
  OpenBLAS (elsewhere). Off by default so the std-lib-only promise holds
  for anyone who doesn't ask for it.

See [note 17](notes/17_performance.md) for the full story, including how
profiling led to each choice and why GPU support is deliberately not on
the roadmap.
