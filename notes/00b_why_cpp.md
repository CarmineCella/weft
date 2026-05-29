# 0b — Why C++?

If you have any background in modern machine learning, the obvious
question on opening this book is: PyTorch exists, NumPy exists, the whole
field has consolidated around Python — why is this library in C++? The
answer is not "C++ is better than Python." It isn't, in general. The
answer is that **C++ is better than Python for what this book is trying
to teach.** The case is worth making honestly, because if you don't buy
it you'll spend the whole book wondering why we're not using PyTorch, and
that's a distraction from the actual material.


## What the book is for

Two goals, which point at the same language:

1. To teach how every piece of a neural network actually works — the
   matrix algebra, the gradients, the optimizers, the convolutions, the
   attention mechanism — at the level of someone who could implement
   each piece themselves, with a pencil if they had to.
2. To build a foundation that's directly usable for deploying neural
   networks in audio applications: VST plugins, audio units, embedded
   sound systems, real-time DSP. All of these are written in C++,
   without exception.

Goal 1 rules out heavy ML frameworks. Goal 2 rules out Python entirely
for the eventual application target. The intersection is "implement
from scratch, in the language audio actually runs in." That's C++.


## Why frameworks hide what we want to teach

If we wrote this library in Python with PyTorch, the entire chapter on
backpropagation would be one line:

```python
loss.backward()
```

That line works. You get correct gradients. But you've learned nothing
about how those gradients were computed. The framework's autograd has
hidden the math.

The same happens at every other level:

- `nn.Linear(in, out)` hides the weight matrix, the bias, and the
  forward pass.
- `nn.Conv2d` hides the im2col-and-matmul trick that makes 2-D
  convolution fast.
- `optim.Adam` hides the per-parameter first and second moments, the
  bias correction, the update rule.
- `nn.MultiheadAttention` hides the queries, keys, values, scaling,
  softmax, and the multi-head projection.

PyTorch is a beautiful productivity tool — but the very things that
make it good for productivity make it bad for understanding. When the
goal is to *demystify* each of these pieces, hiding them defeats the
purpose. Calling a black box doesn't teach what's inside the box.


## What C++ forces you to do

C++ has no autograd, no `nn.Module`, no decorator that quietly records
a computation graph. To do backprop in C++, you have to:

1. Allocate a weight matrix `W` and bias `b`, explicitly, with shapes.
2. Implement the forward pass: `Z = W * X + b`.
3. Cache the input `X` because the backward pass needs it.
4. When the next layer's gradient `dZ` arrives, compute by hand:
   - `dW = dZ * X^T`   (with batch averaging applied in the loss),
   - `db = sum across the batch of dZ`,
   - `dX = W^T * dZ`,
5. Hand `dW` and `db` to the optimizer, which uses them to update `W`
   and `b`.

*That is the actual content of backprop.* The student doesn't learn it
from PyTorch; they learn it from writing it. Once you've written it for
one layer, you've written it for all of them, and you understand the
algorithm permanently.

The same is true of every other piece. Once you've written `im2col`
yourself and seen convolution become a matrix multiply, you'll never
mistake "the conv layer applies a kernel" for the full picture again.
Once you've written Adam's running averages by hand, you'll know
exactly what changes when someone says "we tweaked beta2."


## Why C++ specifically, instead of Python without frameworks?

A fair objection: *"OK, no frameworks — but couldn't you just use Python
with NumPy and avoid the boilerplate of C++?"* Some introductory courses
do exactly that, and it works. The reasons we went the additional step
to C++:

**Type safety as a teaching tool.** In Python, `dense.forward(x)`
accepts anything that responds to multiplication. Pass the wrong shape
and you get a runtime error somewhere deep in the call stack. In C++
with templates, `Dense::forward(const Matrix<float>&)` literally cannot
accept a `Tensor4D<float>` — the compiler refuses. Wrong shape, wrong
rank, wrong scalar type: all caught at the call site, before you even
run the program. For someone learning the data flow of a neural network,
that compile-time enforcement is a feature, not a bureaucracy. It tells
you exactly where the boundaries between data structures are.

**Memory layout becomes visible.** Our `Matrix` stores numbers in a flat
row-major buffer. You SEE that — it's `std::vector<T> data_` and an
indexing formula. Our `Tensor4D` stores them in NCHW order, and the
index formula is right there in the code:
`data[((n*C + c)*H + h)*W + w]`. When you understand WHY this layout
makes 2-D spatial convolution cache-friendly, you've learned something
real about how hardware works. In Python the layout is hidden by NumPy
and you never have to think about it.

**Parallelism becomes visible.** Our matmul parallelises across rows
with `std::thread`. You SEE the threads being spawned and joined. You
SEE the work-threshold below which threading isn't worth its overhead.
You SEE the row-chunking that avoids synchronisation. In Python, threads
are essentially useless because of the GIL, and the real parallelism
happens inside NumPy or PyTorch in code you cannot see or modify.

**The performance story is honest and modifiable.** When this library
trains a network slowly, you can profile it and see exactly where the
time goes — and fix it. We did exactly this when we added the
`make BLAS=1` opt-in: profiling showed `Matrix::operator*` was the
bottleneck, so we optionally route it to a tuned BLAS library via an
`if constexpr` dispatch. The whole story — threads, BLAS dispatch, the
option to leave it off — is right there in `Matrix.h`, observable and
modifiable. In Python with PyTorch, performance is whatever the
framework decides; you can benchmark it, but you can't open it up and
change it.

**C++ is the language audio lives in.** This is a book for music
applications. The eventual goal is for the things you learn here to
make their way into actual audio tools — VST plugins, audio units,
embedded systems, real-time DSP. **All of those are written in C++.**
There is no path from a Python prototype to a JUCE plugin that doesn't
involve rewriting in C++. By starting in C++, you skip that rewrite —
the language you learn the math in IS the language your final
application will run in.

**C++ teaches systems thinking.** Beyond the math, neural networks are
systems: they consume data, allocate memory, run computations in
parallel, share resources, serialise to disk. C++ exposes these
concerns. You make choices about ownership (`unique_ptr` for layers,
raw pointers for transient optimiser state), move semantics, binary
serialisation formats with magic headers, optional dependencies guarded
by macros. These are real engineering concerns that Python hides
almost entirely — and that you eventually need to understand if you
want to build anything that runs outside a notebook.

**C++ has no version churn.** PyTorch's API changes frequently; models
written in 2018 don't always run cleanly in 2024. A Python ML codebase
needs maintenance just to keep running. A C++ codebase using
`-std=c++17` and the standard library will compile and run identically
in 2044. The numerical methods we're teaching don't change. The
language we teach them in shouldn't add noise around them.


## Honest counterarguments

It would be dishonest not to address the costs:

- **More lines of code per layer.** A PyTorch `nn.Linear` is shorter
  than our `Dense.h`. We accept this — the extra lines are the lesson.
- **Slower to iterate.** Every architectural experiment requires a
  recompile. We accept this too; compile time is negligible at our
  scale.
- **No autograd, so we derive every gradient by hand.** This is often
  listed as a downside of C++. For a learning project it is the entire
  point.
- **Less ecosystem.** PyTorch has pretrained models, datasets, model
  hubs, deployment tools. We have none of that. For a finished
  application you'd reach for those eventually; for learning, you
  don't need any of them.

There is also one place where C++ is genuinely worse: for the kind of
large-scale empirical experimentation that drives most modern ML
research (sweeping hyperparameters across hundreds of runs, comparing
dozens of architectures on huge datasets), Python's REPL and notebook
ecosystem are the right tools. We are not doing that kind of research
in this book. We are building a small set of architectures carefully.


## When you would NOT use C++

To be clear: this argument is for this project specifically, not for
machine learning in general. If you are:

- prototyping a new model architecture quickly,
- working with very large datasets that need distributed training,
- using off-the-shelf pretrained models (BERT, Whisper, CLIP),
- writing research papers that need rapid iteration on ideas,
- training in the cloud on GPUs you don't own,

you should reach for Python with PyTorch. That's what PyTorch is for,
and there's no shame in using the right tool for the job.

The argument here is narrower: for **learning** how neural networks
work, and for **deploying** them in **audio applications**, C++ is the
right choice. For research at scale on general datasets, it isn't.


## The point

Frameworks make easy things easy and hard things invisible. For a
learning project, you want the opposite: hard things visible, no easy
magic. C++ gives you that. Every gradient in this library is one you
derived. Every operation is one you wrote. Every optimisation is one
you can profile and change. There is no `loss.backward()` to hide
behind.

When you finish this book, you will have written a complete
neural-network library from scratch in C++ — backprop, conv layers
with im2col, attention with positional encoding, the whole arc. Then
you can use PyTorch if you want to, and you will know exactly what it
is doing under the hood, because you will have built each piece
yourself.
