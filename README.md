# weft

**W**eights **E**stimated **F**rom **T**raining — a small, unpretentious neural-network
library written from scratch in C++, with no external dependencies (standard library only).

This is a learning project: the goal isn't just working code, but understanding every
piece of it. The library is built up one component at a time, and the *why* behind each
component is written down in companion `.md` notes.

## Principles

- **Header-only & templated.** Every class lives in its own capitalized `.h` file
  (`Matrix.h`, `Layer.h`, …), templated on the scalar type (`float` or `double`).
- **No dependencies.** Only the C++ standard library.
- **Readable like the math.** Operator overloading lets a layer's forward pass be written
  as `Z = W * X + b`.
- **One namespace:** everything lives in `namespace weft`.

## Conventions

- A batch of examples is stored **one example per column**: an input batch has shape
  `(features x batch_size)`, a weight matrix has shape `(out x in)`, and the forward pass
  is `Z = W * X + b`.
- Internally, matrix data is stored in a flat, **row-major** array for cache friendliness.

## Build & run the tests

```sh
g++ -std=c++17 -Wall -Wextra test_matrix.cpp -o test_matrix
./test_matrix
```

## Roadmap

1. **Fully-connected classifier** — IRIS, handwritten digits, orchestral
   instrument sounds.
2. **(Variational) autoencoder** — with generative applications for sound.
3. **Convolutional network** — classification on a larger image dataset.

