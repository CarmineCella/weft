# 01 — Matrix: the foundation

Every neural network is, at its heart, a sequence of matrix operations. Before we
build a layer that learns, we need a solid matrix class — and we need to understand
*why* a matrix is the right abstraction, not an arbitrary choice. This note explains
both: first the conceptual leap from a single neuron to a matrix, then the design
decisions inside `src/Matrix.h`.

---

## 1. The conceptual leap: from neuron to matrix

### A neuron is a dot product

Strip an artificial neuron of all the mystique and what's left is embarrassingly
simple. It takes a vector of inputs `x`, has its own weight vector `w` and a bias
scalar `b`, and computes:

```
z = w · x + b = w₁x₁ + w₂x₂ + … + wₙxₙ + b
```

That value `z` is then passed through a nonlinearity (covered in a later note). The
dot product is the *entire* computation of a neuron. Everything else — depth,
learning, representation — is built on top of this one operation.

### A layer is a matrix

A layer isn't one neuron, it's many — say `m` neurons all looking at the same `n`
inputs. Each neuron `i` has its own weight vector `wᵢ`. If we stack those weight
vectors as the **rows** of a matrix `W`, then `W` has shape `(m × n)`: row `i` is
"the incoming weights to neuron `i`."

The whole layer's output is then:

```
z = W x + b           shapes:  (m×n)(n) + (m) = (m)
```

A single matrix–vector product computes *all* the neurons simultaneously.

This is why `W` is shaped `(out × in)` and not the other way around. The shape isn't
a convention — it follows directly from "each output neuron is a weighted sum of the
inputs."

### Batching: why we need matrix × matrix

You rarely push one example through at a time. You push a *batch*. Take `N` input
examples, each a column vector of length `n`, and lay them side by side as the
columns of a matrix `X` of shape `(n × N)`. Then:

```
Z = W X + b           shapes:  (m×n)(n×N) + (m×1) = (m×N)
```

Column `c` of `Z` is the layer output for example `c`. Processing the whole batch
is now a *single* matrix–matrix multiply instead of `N` separate ones.

Why this matters: one big dense multiply uses the CPU cache vastly better than many
tiny ones, and the same operation is what a GPU parallelizes trivially. **Matrix
multiplication is the beating heart of the whole library**, which is why we start
here.

---

## 2. Conventions used in weft

Two related but separate choices shape the whole library:

**Math convention — examples as columns.**
An input batch has shape `(features × batch_size)`. A weight matrix has shape
`(out × in)`. The forward pass is `Z = W·X + b`. This is the convention used in
most "from scratch" textbooks (e.g. Nielsen, *Neural Networks and Deep Learning*).
Frameworks like PyTorch flip it (batch is the first dimension), but the column
convention reads more like the linear algebra and is friendlier for teaching.

**Storage convention — row-major in memory.**
Internally, the numbers live in a single flat array, with element `(i, j)` at
position `i * cols + j`. This is *independent* of the math convention above; it's
just a memory-layout choice for cache friendliness.

---

## 3. Walking through Matrix.h

Most of the choices in `src/Matrix.h` aren't arbitrary. Each one encodes something
about how the rest of the library will feel to use.

### Why templated on the scalar type

```cpp
template <typename T = float>
class Matrix { ... };
```

Neural nets normally run in `float` — half the memory, faster, plenty of precision.
But when we later want to *check* that our backprop gradients are correct (a
numerical gradient check — the single best way to catch backprop bugs), we'll want
`double` for accuracy. Templating lets us flip between them by changing one type.

Templates also satisfy the "all code in `.h`" rule naturally: template definitions
essentially *must* live in headers.

### Why a flat `std::vector`, not vector-of-vectors

A `vector<vector<T>>` scatters each row somewhere random in memory, so iterating it
thrashes the cache. One contiguous block of `rows*cols` floats sits in a
predictable, cache-friendly line. Element access is a single multiply-add to index
— which is exactly what `operator()(i, j)` compiles down to.

### Why the `i-k-j` loop ordering for matmul

```cpp
for (i) for (k) for (j)
    C(i,j) += A(i,k) * B(k,j);
```

The textbook ordering is `i-j-k`, computing one element of `C` at a time. That
stride-walks through `B` column-wise, which is bad for cache because consecutive
accesses jump rows. The `i-k-j` ordering instead has the innermost loop walk
*contiguously* through `B`'s row `k` and `C`'s row `i` simultaneously. Same number
of multiplications, much friendlier to the cache. A free win.

### Why `+` quietly broadcasts a column vector

The forward pass `Z = W·X + b` has `b` of shape `(out × 1)` and `W·X` of shape
`(out × batch)`. For the code to read like the math, `+` needs to add `b` to *every
column* of `W·X`. So `operator+` checks: same shape → element-wise; right operand
is a column vector → broadcast across columns.

Limiting broadcasting to this one case (rather than full NumPy-style broadcasting)
is deliberate. More magic means more surprises; this is the only broadcast a
feed-forward net actually needs.

### Why `apply` is templated on the function

```cpp
template <typename Func>
Matrix apply(Func f) const;
```

Activation functions and their derivatives are all element-wise. Templating on the
callable means the compiler can inline a lambda — no `std::function` indirection,
no virtual call. ReLU will literally be `M.apply([](T x){ return x > 0 ? x : 0; })`.

### Why `hadamard` is a free function

`operator*` already means matrix multiplication. Overloading it for the element-wise
product — which behaves completely differently — would be a trap. In backprop
you'll see *both* on the same line, and they must be visually distinct:

```cpp
auto delta = hadamard(W_next.transpose() * delta_next,
                      activation_derivative);
```

The element-wise product gets its own name precisely so it can't be confused with
the matmul.

---

## 4. Seeing it work: the forward pass falls out

With just `Matrix.h`, the forward pass of a linear layer already reads like the
equation:

```cpp
#include "Matrix.h"
using weft::Matrix;

Matrix<float> W(3, 2);            // 3 output neurons, each looking at 2 inputs
W.randomizeNormal(0.f, 0.1f);

Matrix<float> X{{1, 2, 3, 4},     // feature 1 across 4 examples
                {5, 6, 7, 8}};    // feature 2 across 4 examples  -> (2 x 4)

Matrix<float> b(3, 1, 0.5f);      // one bias per output neuron

Matrix<float> Z = W * X + b;      // forward pass of a Dense layer
                                  // Z has shape (3 x 4): one column per example
```

The dimensions check themselves: multiply a `(3 × 2)` by a `(2 × 4)` and you get
`(3 × 4)` — the inner `2`s cancel. The bias `(3 × 1)` broadcasts across columns.
And the line `Z = W * X + b` is *literally* the equation we derived above.

That correspondence between the code and the math is the thing we're protecting at
every step in this library.

---

## 5. What's next

The forward pass tells you what a layer *computes*, but not how it *learns*. The
next note covers **backpropagation** — the chain rule applied systematically — and
`src/Layer.h` builds the first learnable layer (`Dense`) on top of `Matrix`.
