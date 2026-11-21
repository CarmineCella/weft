# 1 — Why C++?

Readers with a background in modern machine learning will ask why this
library is in C++, when PyTorch and NumPy exist and the field has
consolidated around Python. C++ is not better than Python in general.
It is better than Python for what this book is trying to teach. This
chapter makes that case, so that the choice of language does not
distract from the material later.


## What the book is for

The goals of the book point at the same language:

1. To teach how every piece of a neural network works (the matrix
   algebra, the gradients, the optimizers, the convolutions, the
   attention mechanism) at the level of someone who could implement
   each piece themselves, with a pencil if they had to.
2. To build a foundation that is directly usable for deploying neural
   networks in audio applications: VST plugins, audio units, embedded
   sound systems, real-time DSP. All of these are written in C++,
   without exception.

Goal 1 rules out heavy ML frameworks. Goal 2 rules out Python for the
eventual application target. What remains is to implement everything
from scratch, in the language audio software runs in, which is C++.


## Why frameworks hide what we want to teach

If we wrote this library in Python with PyTorch, the entire chapter on
backpropagation would be one line:

```python
loss.backward()
```

That line works and gives correct gradients, but it teaches nothing
about how those gradients are computed. The framework's autograd hides
the math.

The same happens at every other level:

- `nn.Linear(in, out)` hides the weight matrix, the bias, and the
  forward pass.
- `nn.Conv2d` hides the im2col-and-matmul trick that makes 2-D
  convolution fast.
- `optim.Adam` hides the per-parameter first and second moments, the
  bias correction, the update rule.
- `nn.MultiheadAttention` hides the queries, keys, values, scaling,
  softmax, and the multi-head projection.

PyTorch is a good productivity tool, but the features that make it
productive make it poor for understanding. Our goal is to explain each
of these pieces, and hiding them defeats that goal.


## What C++ requires

C++ has no autograd, no `nn.Module`, no decorator that records a
computation graph. To do backprop in C++, you have to:

1. Allocate a weight matrix `W` and bias `b`, explicitly, with shapes.
2. Implement the forward pass: `Z = W * X + b`.
3. Cache the input `X` because the backward pass needs it.
4. When the next layer's gradient `dZ` arrives, compute by hand:
   - `dW = dZ * X^T`   (with batch averaging applied in the loss),
   - `db = sum across the batch of dZ`,
   - `dX = W^T * dZ`,
5. Hand `dW` and `db` to the optimizer, which uses them to update `W`
   and `b`.

These steps are the content of backprop. A student learns them by
writing them, not by calling PyTorch. The steps are the same for every
layer, so writing them for one layer is enough to understand the
algorithm.

The same is true of every other piece. Once you have written `im2col`
yourself and seen convolution become a matrix multiply, you know that
"the conv layer applies a kernel" is not the full description. Once you
have written Adam's running averages by hand, you know what changes
when someone says "we tweaked beta2."


## Why C++ instead of Python without frameworks

One could avoid frameworks and still use Python with NumPy, without the
boilerplate of C++. Some introductory courses do that, and it works. We
went the additional step to C++ for the following reasons.

**Type safety as a teaching tool.** In Python, `dense.forward(x)`
accepts anything that responds to multiplication. Pass the wrong shape
and you get a runtime error somewhere deep in the call stack. In C++
with templates, `Dense::forward(const Matrix<float>&)` cannot accept a
`Tensor4D<float>`: the compiler refuses. A wrong rank or wrong scalar
type is caught at the call site, before the program runs. (A wrong
shape is caught at run time, by an explicit check that throws at the
offending operation.) For someone learning the data flow of a neural
network, this compile-time enforcement is useful. It shows where the
boundaries between data structures are.

**Memory layout becomes visible.** Our `Matrix` stores numbers in a flat
row-major buffer. This is visible in the code: it is
`std::vector<T> data_` and an indexing formula. Our `Tensor4D` stores
them in NCHW order, and the index formula is in the code:
`data[((n*C + c)*H + h)*W + w]`. Understanding why this layout makes
2-D spatial convolution cache-friendly teaches something about how
hardware works. In Python the layout is hidden by NumPy and you never
have to think about it.

**Parallelism becomes visible.** Our matmul parallelises across rows
with `std::thread`. The code shows the threads being spawned and
joined, the work threshold below which threading is not worth its
overhead, and the row-chunking that avoids synchronisation. In Python,
threads are essentially useless because of the GIL, and the real
parallelism happens inside NumPy or PyTorch in code you cannot see or
modify.

**Performance is visible and modifiable.** When this library trains a
network slowly, you can profile it, see where the time goes, and fix
it. We did this when we added the `make BLAS=1` opt-in: profiling
showed `Matrix::operator*` was the bottleneck, so we optionally route
it to a tuned BLAS library via an `if constexpr` dispatch. The threads,
the BLAS dispatch and the option to leave it off are all in `Matrix.h`,
where you can read and change them. In Python with PyTorch, performance
is whatever the framework decides. You can benchmark it, but you cannot
open it and change it.

**C++ is the language of audio software.** This is a book for music
applications. The eventual goal is for what you learn here to be used
in audio tools: VST plugins, audio units, embedded systems, real-time
DSP. All of those are written in C++. There is no path from a Python
prototype to a JUCE plugin that doesn't involve rewriting in C++.
Starting in C++ avoids that rewrite, because the language you learn the
math in is the language your final application runs in.

**C++ teaches systems thinking.** Beyond the math, neural networks are
systems: they consume data, allocate memory, run computations in
parallel, share resources, serialise to disk. C++ exposes these
concerns. You make choices about ownership (`unique_ptr` for layers,
raw pointers for transient optimiser state), move semantics, binary
serialisation formats with magic headers, optional dependencies guarded
by macros. Python hides these engineering concerns almost entirely, and
you need to understand them to build anything that runs outside a
notebook.

**C++ is stable across versions.** PyTorch's API changes frequently;
models written in 2018 don't always run cleanly in 2024. A Python ML
codebase needs maintenance to keep running. A C++ codebase using
`-std=c++17` and the standard library will compile and run identically
in 2044. The numerical methods we teach don't change, and the language
we teach them in should not add changes of its own.


## Costs

The choice has costs:

- **More lines of code per layer.** A PyTorch `nn.Linear` is shorter
  than our `Dense.h`. We accept this, because the extra lines are what
  the book teaches.
- **Slower to iterate.** Every architectural experiment requires a
  recompile. We accept this too; compile time is negligible at our
  scale.
- **No autograd, so we derive every gradient by hand.** This is often
  listed as a downside of C++. For a learning project it is the
  purpose.
- **Less ecosystem.** PyTorch has pretrained models, datasets, model
  hubs, deployment tools. We have none of that. For a finished
  application you would eventually use those; for learning, you don't
  need any of them.

There is also one place where C++ is worse: for the kind of
large-scale empirical experimentation that drives most modern ML
research (sweeping hyperparameters across hundreds of runs, comparing
dozens of architectures on huge datasets), Python's REPL and notebook
ecosystem are the right tools. We are not doing that kind of research
in this book. We are building a small set of architectures carefully.


## When you would not use C++

This argument is for this project specifically, not for machine
learning in general. If you are:

- prototyping a new model architecture quickly,
- working with very large datasets that need distributed training,
- using off-the-shelf pretrained models (BERT, Whisper, CLIP),
- writing research papers that need rapid iteration on ideas,
- training in the cloud on GPUs you don't own,

you should use Python with PyTorch. That is what PyTorch is for.

The argument here is narrower: for learning how neural networks work,
and for deploying them in audio applications, C++ is the right choice.
For research at scale on general datasets, it isn't.


## What you will have at the end

In this library every gradient is derived by hand, every operation is
written out, and every optimisation can be profiled and changed. No
step is hidden behind a call such as `loss.backward()`.

When you finish this book, you will have written a complete
neural-network library from scratch in C++, including backprop, conv
layers with im2col, and attention with positional encoding. You can
then use PyTorch if you want to, and you will know what it does
internally, because you will have built each piece yourself.


## Further reading

PyTorch [@paszke2019pytorch] and NumPy [@harris2020numpy] are the tools
this book deliberately does not use. What a framework's automatic
differentiation does behind `loss.backward()` is surveyed in
[@baydin2018autodiff]. For a broad textbook treatment of everything the
following chapters build by hand, see [@goodfellow2016deep].
