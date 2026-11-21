# 21 — From matrices to tensors

For twenty chapters we have used a single data structure, `Matrix`, for
all of the numerical work in this library. Every layer in every network
(the MLPs, the autoencoders, the variational autoencoders) has taken
a matrix in and produced a matrix out. That has been sufficient because
all of those networks work on flat feature vectors: an MNIST image is
784 numbers, an audio frame is 2049 numbers, and a batch of them is a
two-dimensional arrangement (features down, examples across), with one
row index and one column index.

Convolutional layers do not fit that shape. A convolutional layer's
input is a batch of images, and an image is 2-D in space (height and
width). It also has a *channel* dimension: one for grayscale, three
for RGB, sixteen or thirty-two for the intermediate feature maps
deeper in a network. So the natural shape is **four-dimensional**:
batch × channels × height × width. We need a data structure that holds
that shape.

This chapter describes that data structure, the layout we chose for
it, and why we built `Tensor4D` as its own class rather than
generalising `Matrix` to N dimensions.


## What a tensor is

In mathematics a *tensor* is a multi-dimensional array of numbers, a
generalisation of objects we already know:

- A **scalar** is a 0-D tensor: a single number.
- A **vector** is a 1-D tensor: an array indexed by one integer.
- A **matrix** is a 2-D tensor: a grid indexed by two integers.
- An **image batch** is a 4-D tensor: indexed by
  (batch, channel, height, width).

(The word "tensor" has a deeper mathematical meaning in physics and
differential geometry, which machine learning uses only loosely. In
ML, *tensor* mostly means "N-dimensional array of numbers.")

So `Matrix` is mathematically a 2-tensor with shape `(rows, cols)`,
and our new `Tensor4D` is a 4-tensor with shape `(N, C, H, W)`. Since
`Matrix` is a special case of tensor, a single unified `Tensor` class
could cover both. Whether it should is the design question of this
chapter, and the answer is a tradeoff. We come back to it after
settling the layout.


## NCHW: the order of the axes

A 4-D tensor with shape `(N, C, H, W)` could be stored in memory in
any of 24 different orderings. Two are common in practice:

- **NCHW**: batch, channels, height, width  (outermost first).
- **NHWC**: batch, height, width, channels.

The choice determines which dimension is innermost in memory, i.e.,
which adjacent indices correspond to adjacent bytes. With NCHW (what
we use), walking the flat storage one element at a time walks `W`
fastest (within one row), then `H` (within one channel of one image),
then `C` (within one image), then `N` (between images). In our
`Tensor4D` the element at index `(n, c, h, w)` is at flat offset

    ((n * C + c) * H + h) * W + w

So pixels in the same row of the same channel of the same image sit
next to each other in memory.

This suits a 2-D spatial convolution, which slides a kernel across
the H-W plane of one channel at a time. Having that plane contiguous
in memory means the inner loop walks contiguous bytes, which is good
for the CPU cache and for BLAS. NHWC interleaves channels (pixels of
different channels at the same `(h, w)` position are adjacent), which
is better for some quantisation techniques and for memory-bound
channel-wise operations, but less natural for spatial convolution on
CPU.

PyTorch and cuDNN default to NCHW; we follow the same convention so
the code reads as expected to anyone arriving from that ecosystem.


## One class or two

The alternatives for the design question are the following.

**Path A: one unified Tensor.** A single class that holds an
N-dimensional shape and a flat data buffer, with operations that work
for any rank. This is what every production library does: PyTorch's
`torch.Tensor`, NumPy's `ndarray`, TensorFlow's `tf.Tensor`, JAX's
arrays. There is one object, and a 2-D matrix is a tensor with rank 2.

**Path B: separate Matrix and Tensor4D.** This is what we did.
`Matrix` is its own class with its own operations (`operator*`,
`transpose`, `sumColumns`). `Tensor4D` is its own class. They are
connected by explicit free functions like `flatten(Tensor4D) → Matrix`
when data needs to move between them.

Each has advantages and disadvantages, and which one is right depends
on what the library is for.


## The case for Path A (unified Tensor)

- **Mathematical cleanliness.** A matrix is a tensor of rank 2;
  having one class matches the math.
- **One API to learn.** Users construct one kind of object regardless
  of dimensionality.
- **`flatten` becomes `reshape`.** No type conversion is needed; the
  same storage is reinterpreted with a different shape. The data
  buffer never moves.
- **Future-proof.** Whatever rank comes up next (3-D for attention
  sequences, 5-D for video), the class already handles it. No new
  type is needed each time.
- **Matches what students will use professionally.** Anyone arriving
  here from PyTorch already thinks in those terms; anyone going to
  PyTorch later won't have to unlearn anything.


## The case for Path B (what we have)

- **Type safety.** Our `Dense::forward(Matrix)` cannot accept a 4-D
  tensor; the compiler refuses. Passing the wrong-rank object to a
  layer is a compile error, not a runtime exception deep inside the
  function. This matters in a learning context, where the aim is to
  build a mental model of the data flow: mistakes are caught at the
  boundary.

- **Operation specificity.** Many of `Matrix`'s operations only make
  sense in 2-D. A "transpose" of a 4-D tensor has to say which two
  axes are swapped. A "matrix multiply" of two 4-D tensors has to say
  which pair of axes is contracted. In a unified Tensor each of these
  needs either restriction to rank 2 (special cases throughout the
  code) or generalisation to indexed-axis operations (a different and
  more complex API). The two-class design lets each class have the
  operations that fit its dimensionality, with no special-casing.

- **Pedagogical clarity.** When you read

      Z  = W * X + b
      dW = dZ * X.transpose()

  in our `Dense` layer, the code reads like the matrix-form backprop
  equations from chapter 3. There is no `.view()`, no
  `.contract(...)`, and there are no axis names. The math and the code
  use the same symbols. In a unified Tensor every operation must
  specify which axes are involved, which moves the code one step away
  from its mathematical form.

- **Small, focused files.** `Matrix.h` is about 320 lines and can be
  read in one sitting; `Tensor4D.h` is 160. A unified `Tensor` with
  proper N-D indexing, broadcasting, reshape, and generalised
  contractions is easily a thousand lines, and the reader can no
  longer keep all of it in mind while reading conv code.

- **An explicit conversion step.** The explicit `flatten` step
  between the conv stack and the dense classifier shows where the
  architecture switches from operating on spatial structure to
  operating on flattened feature vectors. In PyTorch that line is
  `x = x.view(N, -1)` and the change is not visible in the types. In
  our code the type changes, `Tensor4D` becomes `Matrix`, and the
  reader sees that the architecture has changed at that point.


## Why production libraries chose Path A

The libraries that ML engineers use every day chose Path A, and they
are right to. Their reasons correspond to the places where their goals
differ from ours:

- They are libraries for working practitioners, not learners. Their
  type system is not meant to teach; it is meant not to get in the
  way.
- They serve every possible rank (1-D vectors, 3-D sequences, 5-D
  video, dynamically-shaped data) and cannot afford one class per
  rank.
- They optimise broadcasting and reshape heavily, so a `.view()` call
  is essentially free and idiomatic.
- They have automatic differentiation, which records computational
  graphs over generic tensors and would be much more complicated with
  per-rank types.

For those goals, unified is the correct choice. For ours, teaching,
separate is correct.


## The design principle

A data structure gets its own type when it has its own conceptual
identity. Matrix has one: it is the object on which the backprop math
is written. Tensor4D has one: it is what a batch of feature maps looks
like, and the operations on it (convolution, pooling) are different
in kind from matrix operations. If a third structure came along that
was only a matrix with a bigger shape, we would not necessarily need a
new class for it.

When attention arrives in [chapter 25](25_attention.md), the natural
data shape would be 3-D (batch × time × features), and the same
question arises: a new `Tensor3D` class, or `Matrix` again. There we
avoid a third type by processing one sequence at a time, so that a
sequence is a `Matrix` with one position per column.

If this library ever grew into a more general framework (N-D tensors,
broadcasting, autograd, GPU support), refactoring toward a unified
`Tensor` would be a natural step. The current design is a deliberate
choice for an educational stage, not a mistake to be undone, and a
more general library would eventually outgrow it.

The next chapter builds `Conv2D`, the layer that uses the Tensor4D. It
turns a `Tensor4D(N, in_C, H, W)` into a `Tensor4D(N, out_C, H', W')`
with the im2col-and-matmul approach.


## Further reading {#sec-21-from-matrices-to-tensors-further-reading}

The NCHW layout follows the convention of PyTorch [@paszke2019pytorch].
The convolutional architectures that make a four-dimensional tensor
necessary are those of [@lecun1998gradient].
