# 2 — Matrix: the foundation

Every neural network is a sequence of matrix operations. Before we
build a layer that learns, we need a matrix class, and we need to
understand why a matrix is the right abstraction. This chapter explains
both: first the step from a single neuron to a matrix, then the design
decisions inside `src/Matrix.h`.

---

## 1. From neuron to matrix

### A neuron is a dot product

An artificial neuron is simple. It takes a vector of inputs `x`, has
its own weight vector `w` and a bias scalar `b`, and computes:

```
z = w · x + b = w₁x₁ + w₂x₂ + … + wₙxₙ + b
```

That value `z` is then passed through a nonlinearity (covered in a
later chapter). The dot product is the whole computation of a neuron.
Depth, learning and representation are all built on this one operation.

### A layer is a matrix

A layer has many neurons, say `m` neurons all looking at the same `n`
inputs. Each neuron `i` has its own weight vector `wᵢ`. If we stack
those weight vectors as the **rows** of a matrix `W`, then `W` has
shape `(m × n)`: row `i` holds the incoming weights to neuron `i`.

The whole layer's output is then:

```
z = W x + b           shapes:  (m×n)(n) + (m) = (m)
```

A single matrix–vector product computes all the neurons at once.

This is why `W` is shaped `(out × in)` and not the other way around.
The shape follows from "each output neuron is a weighted sum of the
inputs."

### Batching: why we need matrix × matrix

We rarely push one example through at a time. We push a batch. Take `N`
input examples, each a column vector of length `n`, and lay them side
by side as the columns of a matrix `X` of shape `(n × N)`. Then:

```
Z = W X + b           shapes:  (m×n)(n×N) + (m×1) = (m×N)
```

Column `c` of `Z` is the layer output for example `c`. Processing the
whole batch is now a single matrix–matrix multiply instead of `N`
separate ones.

This matters because one large dense multiply uses the CPU cache much
better than many small ones, and it is the same operation that a GPU
parallelizes easily. Matrix multiplication is the central operation of
the library, which is why we start here.

---

## 2. Conventions used in weft

The library fixes a math convention and a storage convention. They are
related but separate.

**Math convention: examples as columns.**
An input batch has shape `(features × batch_size)`. A weight matrix has
shape `(out × in)`. The forward pass is `Z = W·X + b`. This is the
convention used in most "from scratch" textbooks (e.g. Nielsen, *Neural
Networks and Deep Learning*). Frameworks like PyTorch flip it (batch is
the first dimension), but the column convention reads more like the
linear algebra and is easier for teaching.

**Storage convention: row-major in memory.**
Internally, the numbers are stored in a single flat array, with element
`(i, j)` at position `i * cols + j`. This is independent of the math
convention above. It is a memory-layout choice made for cache
efficiency.

---

## 3. Design decisions in Matrix.h

Most of the choices in `src/Matrix.h` have a reason, and each one
affects how the rest of the library is used.

### Why templated on the scalar type

```cpp
template <typename T = float>
class Matrix { ... };
```

Neural nets normally run in `float`, which takes half the memory, is
faster, and has enough precision. Later we want to check that our
backprop gradients are correct with a numerical gradient check, the
best way to catch backprop bugs, and for that we want `double` for
accuracy. Templating lets us switch between them by changing one type.

Templates also fit the "all code in `.h`" rule: template definitions
essentially must live in headers.

### Why a flat `std::vector`, not vector-of-vectors

A `vector<vector<T>>` places each row at an unrelated location in
memory, so iterating over it causes many cache misses. One contiguous
block of `rows*cols` floats is laid out predictably and is
cache-friendly. Element access needs one multiply-add to compute the
index, which is what `operator()(i, j)` compiles down to.

### Why the `i-k-j` loop ordering for matmul

```cpp
for (i) for (k) for (j)
    C(i,j) += A(i,k) * B(k,j);
```

The textbook ordering is `i-j-k`, computing one element of `C` at a
time. That walks through `B` column-wise, which is bad for the cache
because consecutive accesses jump rows. With the `i-k-j` ordering the
innermost loop walks contiguously through `B`'s row `k` and `C`'s row
`i` at the same time. The number of multiplications is the same and
the cache behaviour is much better, at no extra cost.

### Why `+` broadcasts a column vector

The forward pass `Z = W·X + b` has `b` of shape `(out × 1)` and `W·X`
of shape `(out × batch)`. For the code to read like the math, `+` needs
to add `b` to every column of `W·X`. So `operator+` checks the shapes:
if they are the same, it adds element-wise; if the right operand is a
column vector, it broadcasts across columns.

Limiting broadcasting to this one case (rather than full NumPy-style
broadcasting) is deliberate. More implicit behaviour means more
surprises, and this is the only broadcast a feed-forward net needs.

### Why `apply` is templated on the function

```cpp
template <typename Func>
Matrix apply(Func f) const;
```

Activation functions and their derivatives are all element-wise.
Templating on the callable means the compiler can inline a lambda, with
no `std::function` indirection and no virtual call. ReLU will be
`M.apply([](T x){ return x > 0 ? x : 0; })`.

### Why `hadamard` is a free function

`operator*` already means matrix multiplication. Overloading it for the
element-wise product, which behaves completely differently, would
invite mistakes. In backprop both appear on the same line, and they
must be visually distinct:

```cpp
auto delta = hadamard(W_next.transpose() * delta_next,
                      activation_derivative);
```

The element-wise product has its own name so that it cannot be confused
with the matmul.

---

## 4. The forward pass in code

With only `Matrix.h`, the forward pass of a linear layer already reads
like the equation:

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

The dimensions can be checked directly: multiply a `(3 × 2)` by a
`(2 × 4)` and you get `(3 × 4)`, because the inner `2`s cancel. The
bias `(3 × 1)` broadcasts across columns. The line `Z = W * X + b` is
the equation we derived above.

We keep this correspondence between the code and the math throughout
the library.

---

## 5. From the forward pass to learning

The forward pass says what a layer computes, but not how it learns. The
next chapter covers **backpropagation**, the chain rule applied
systematically, and `src/Layer.h` and `src/Dense.h` build the first
learnable layer (`Dense`) on top of `Matrix`.


## Further reading {#sec-02-matrix-further-reading}

The linear algebra used throughout is covered in chapter 2 of
[@goodfellow2016deep]. Why the order of the three loops in a matrix
product matters so much, and how tuned libraries go much further than
our `i-k-j` loop, is explained in [@goto2008anatomy].
