# 1 — Why C++?

This book uses a small C++ library to investigate neural networks for
sound. We write the matrix operations, derive the gradients, and follow
the data from a waveform to a model and back to audible output. C++ lets
us keep that work close to the memory layout and DSP code an audio
programmer already uses.

The important teaching choice is to implement the operations ourselves.
We could do that in another language. Here we use C++17 and the standard
library so that the implementation remains small enough to read and the
default build has few dependencies.


## What the book is for

There are two goals:

1. To teach how every piece of a neural network works (the matrix
   algebra, the gradients, the optimizers, the convolutions, the
   attention mechanism) at the level of someone who could implement
   each piece themselves, with a pencil if they had to.
2. To connect those operations to audio applications: instrument
   recognition, reconstruction, timbre manipulation, synthesis, and
   musical generation.

C++ is common in plugin and embedded audio development, which makes it
a useful setting for these experiments. The programs in this book are
offline CPU applications. Using the same language as an audio host does
not make them ready for an audio callback: allocation, latency, and
threading would need separate measurement and engineering.


## Why implement the operations ourselves?

In PyTorch, a training program can ask for gradients with:

```python
loss.backward()
```

That interface is useful once we trust the operations below it. Our
question is how those operations work. For each convenience a framework
provides, there is something we want to inspect:

- `nn.Linear(in, out)` leads us to the weight matrix, bias, and their
  gradients.
- `nn.Conv2d` leads us to the im2col-and-matmul implementation used here.
- `optim.Adam` leads us to running moments, bias correction, and the
  update rule.
- `nn.MultiheadAttention` leads us to queries, keys, values, scaling,
  softmax, and the output projection.

Framework source is available to study and modify, and frameworks are
useful for checking a small implementation. We build a separate library
to keep the relevant path short: from an equation to a loop, then to a
numerical test.


## What we implement by hand

The C++ standard library has no automatic differentiation, although
libraries such as PyTorch provide it for C++. We deliberately leave it
out of weft. For a dense layer, that means we:

1. Allocate a weight matrix `W` and bias `b`, explicitly, with shapes.
2. Implement the forward pass: `Z = W * X + b`.
3. Cache the input `X` because the backward pass needs it.
4. When the next layer's gradient `dZ` arrives, compute by hand:
   - `dW = dZ * X^T`   (with batch averaging applied in the loss),
   - `db = sum across the batch of dZ`,
   - `dX = W^T * dZ`,
5. Hand `dW` and `db` to the optimizer, which uses them to update `W`
   and `b`.

These steps expose the chain rule and the state a backward pass needs.
Later layers use the same pattern, but each has its own derivative and
shape conventions to check. Implementing one layer gives us a method for
approaching the next.

The same is true of every other piece. Once you have written `im2col`
yourself and seen convolution become a matrix multiply, you know that
"the conv layer applies a kernel" is not the full description. Once you
have written Adam's running averages by hand, you know what changes
when someone says "we tweaked beta2."


## Why C++ instead of Python without frameworks

One could avoid frameworks and still use Python with NumPy, without the
boilerplate of C++. Some introductory courses do that, and it works. We
went the additional step to C++ for the following reasons.

**Visible data boundaries.** In our C++ interface,
`Dense::forward(const Matrix<float>&)` cannot accept a
`Tensor4D<float>`: the compiler refuses. A wrong rank or wrong scalar
type is caught before the program runs. Matrix dimensions remain runtime
values, so dimensional compatibility still needs runtime checks. The
Matrix/Tensor4D distinction makes the boundary between dense and
convolutional operations explicit.

**Memory layout becomes visible.** Our `Matrix` stores numbers in a flat
row-major buffer. This is visible in the code: it is
`std::vector<T> data_` and an indexing formula. Our `Tensor4D` stores
them in NCHW order, and the index formula is in the code:
`data[((n*C + c)*H + h)*W + w]`. Following how the convolution loops
visit that buffer helps explain their cache behavior.
NumPy also exposes shape, strides, and storage order; here we implement
the indexing ourselves.

**Parallelism becomes visible.** Our matmul parallelises across rows
with `std::thread`. The code shows the threads being spawned and
joined, the work threshold below which threading is not worth its
overhead, and the row-chunking that avoids overlapping writes. Threading
and tuned numerical kernels are also available through Python tools.
Here the scheduling decision is part of the implementation we read.

**Performance is visible and modifiable.** When this library trains a
network slowly, you can profile it, see where the time goes, and fix
it. We did this when we added the `make BLAS=1` opt-in: profiling
showed `Matrix::operator*` was the bottleneck, so we optionally route
it to a tuned BLAS library via an `if constexpr` dispatch. The threads,
the BLAS dispatch and the option to leave it off are all in `Matrix.h`,
where you can read and change them. The small implementation makes it
practical to connect a profile to a particular loop.

**C++ teaches systems thinking.** Beyond the math, neural networks are
systems: they consume data, allocate memory, run computations in
parallel, share resources, serialise to disk. C++ exposes these
concerns. You make choices about ownership (`unique_ptr` for layers,
raw pointers for transient optimiser state), move semantics, binary
serialisation formats with magic headers, optional dependencies guarded
by macros. These concerns exist in other languages too; keeping them in
this small codebase lets us examine them alongside the mathematics.

**A small dependency surface.** The default build uses C++17 and the
standard library; a tuned BLAS backend is optional. This reduces the
number of packages needed to run an example. Reproducibility still
depends on compiler settings, hardware, random initialization, and data.
A language standard alone cannot guarantee identical numerical results
across machines or future toolchains.


## Costs

The choice has costs:

- **More lines of code per layer.** A PyTorch `nn.Linear` is shorter
  than our `Dense.h`. We accept this, because the extra lines are what
  the book teaches.
- **Slower to iterate.** Every architectural experiment requires a
  recompile. We accept this too; compile time is negligible at our
  scale.
- **Manual derivatives.** Every new operation needs a backward pass and
  a numerical check. This is a choice of this project, and it creates
  work whenever we extend a model.
- **A limited toolset.** Weft has no GPU backend, automatic
  differentiation, or pretrained-model ecosystem. The examples are
  deliberately small enough for CPU experiments.

For rapid architecture changes, pretrained models, or large training
runs, an established framework offers capabilities we do not reproduce.
Python and NumPy also provide a good setting for learning the same
mathematics. The reason to follow this implementation is the combination
of explicit gradients, systems details, and audio experiments.


## What you will have at the end

In this library every gradient is derived by hand, every operation is
written out, and every optimisation can be profiled and changed. No
step is hidden behind a call such as `loss.backward()`.

By the end, those operations support several ways of representing sound:
features for recognition, spectra for reconstruction, controls for a
synthesiser, symbolic notes, and sparse time-frequency atoms. Comparing
what those representations preserve and lose is as important as getting
the networks to train.


## Further reading {#sec-01-why-cpp-further-reading}

PyTorch [@paszke2019pytorch] and NumPy [@harris2020numpy] are the tools
this book deliberately does not use. What a framework's automatic
differentiation does behind `loss.backward()` is surveyed in
[@baydin2018autodiff]. For a broad textbook treatment of everything the
following chapters build by hand, see [@goodfellow2016deep].
