# 18 — From matrices to tensors

For seventeen notes we have used a single data structure, `Matrix`, for
all of the numerical work in this library. The MLPs, the autoencoders,
the variational autoencoders — every layer in every network has taken
a matrix in and produced a matrix out. That has been sufficient because
all of those networks work on *flat* feature vectors: an MNIST image is
784 numbers, an audio frame is 2049 numbers, and a batch of them is a
two-dimensional arrangement of (features down, examples across). One
row index, one column index. The matrix fit the job.

Convolutional layers do not fit that shape. A convolutional layer's
input is a *batch of images*, and an image is fundamentally 2-D in
space (height and width). It also has a *channel* dimension — one for
grayscale, three for RGB, sixteen or thirty-two for the intermediate
feature maps deeper in a network. So the natural shape is
**four-dimensional**: batch × channels × height × width. We need a
data structure that holds that shape.

This note is about what such a data structure is, what layout we chose
for it, and — because it is a question worth being honest about — why
we built `Tensor4D` as its own class rather than generalising `Matrix`
to N dimensions.


## What a tensor is

In mathematics a *tensor* is a multi-dimensional array of numbers, a
straightforward generalisation of objects we already know:

- A **scalar** is a 0-D tensor: a single number.
- A **vector** is a 1-D tensor: an array indexed by one integer.
- A **matrix** is a 2-D tensor: a grid indexed by two integers.
- An **image batch** is a 4-D tensor: indexed by
  (batch, channel, height, width).

(The word "tensor" carries some deeper mathematical baggage from
physics and differential geometry that machine learning borrowed
lightly. In ML, *tensor* mostly just means "N-dimensional array of
numbers.")

So `Matrix` is mathematically a 2-tensor with shape `(rows, cols)`,
and our new `Tensor4D` is a 4-tensor with shape `(N, C, H, W)`. The
natural question follows immediately: if `Matrix` is a special case of
tensor, why do we have two classes? Couldn't a single unified `Tensor`
cover both? That is the design question this note is really about,
and the answer is a real tradeoff rather than an obvious one. We come
back to it after settling the layout choice.


## NCHW: the order of the axes

A 4-D tensor with shape `(N, C, H, W)` could be stored in memory in
any of 24 different orderings. Two are common in practice:

- **NCHW** — batch, channels, height, width  (outermost first).
- **NHWC** — batch, height, width, channels.

The choice is about which dimension is *innermost in memory* — i.e.,
which adjacent indices correspond to adjacent bytes. With NCHW (what
we use), walking the flat storage one element at a time walks `W`
fastest (within one row), then `H` (within one channel of one image),
then `C` (within one image), then `N` (between images). Concretely,
in our `Tensor4D` the element at index `(n, c, h, w)` lives at flat
offset

    ((n * C + c) * H + h) * W + w

So pixels in the same row of the same channel of the same image sit
next to each other in memory.

That is exactly what a 2-D spatial convolution wants: it slides a
kernel across the H-W plane of one channel at a time. Having that
plane contiguous in memory means the inner loop walks contiguous
bytes, which is friendly to the CPU cache and to BLAS. NHWC interleaves
channels — pixels of different channels at the same `(h, w)` position
are adjacent — which is better for some quantisation tricks and for
memory-bound *channel*-wise operations, but less natural for spatial
convolution on CPU.

PyTorch and cuDNN default to NCHW; we follow the same convention so
the code reads as expected to anyone arriving from that ecosystem.


## One class or two?

Now the design question. There are two reasonable approaches:

**Path A — one unified Tensor.** A single class that holds an
N-dimensional shape and a flat data buffer, with operations that work
for any rank. This is what every production library does: PyTorch's
`torch.Tensor`, NumPy's `ndarray`, TensorFlow's `tf.Tensor`, JAX's
arrays. There is exactly one object, and a 2-D matrix is a tensor
with rank 2.

**Path B — separate Matrix and Tensor4D.** What we did. `Matrix` is
its own class with its own operations (`operator*`, `transpose`,
`sumColumns`). `Tensor4D` is its own class. They are bridged by
explicit free functions like `flatten(Tensor4D) → Matrix` when data
needs to move between them.

The pros and cons cut both ways, and which one is "right" depends on
what the library is for.


## The case for Path A (unified Tensor)

- **Mathematical cleanliness.** A matrix really is a tensor of rank 2;
  having one class matches the math.
- **One API to learn.** Users construct one kind of object regardless
  of dimensionality.
- **`flatten` becomes `reshape`.** No type conversion is needed; you
  just reinterpret the same storage with a different shape. The data
  buffer never moves.
- **Future-proof.** Whatever rank comes up next — 3-D for attention
  sequences, 5-D for video — the class already handles it. No new
  type to invent each time.
- **Matches what students will use professionally.** Anyone arriving
  here from PyTorch already thinks in those terms; anyone going to
  PyTorch later won't have to unlearn anything.


## The case for Path B (what we have)

- **Type safety.** Our `Dense::forward(Matrix)` literally cannot
  accept a 4-D tensor. The compiler refuses. Passing the wrong-rank
  object to a layer is a *compile* error, not a runtime exception
  somewhere deep inside the function. For a learning context — where
  building mental models of the data flow is the whole point — this
  matters; mistakes are caught at the boundary, not buried.

- **Operation specificity.** Many of `Matrix`'s operations only make
  sense in 2-D. What is "transpose" on a 4-D tensor? You have to say
  *which two axes* you're swapping. What is "matrix multiply" on two
  4-D tensors? A contraction over which pair of axes? In a unified
  Tensor each of these needs either restriction to rank 2 (special
  cases sprinkled throughout) or generalisation to indexed-axis
  operations (a different and more complex API). The two-class design
  lets each class have exactly the operations that fit its
  dimensionality, with no special-casing.

- **Pedagogical clarity.** When you read

      Z  = W * X + b
      dW = dZ * X.transpose() / N

  in our `Dense` layer, the code reads literally like the
  matrix-form backprop equations from note 02. No `.view()`, no
  `.contract(...)`, no axis names. The math and the code are the same
  string of symbols. In a unified Tensor every operation must specify
  which axes are involved, which moves the code one step away from
  its mathematical form.

- **Small, focused files.** `Matrix.h` is about 250 lines and can be
  read in one sitting; `Tensor4D.h` is 160. A unified `Tensor` with
  proper N-D indexing, broadcasting, reshape, and generalised
  contractions is easily a thousand lines, and the reader can no
  longer hold all of it in their head while reading conv code.

- **The friction itself is the lesson.** The explicit `flatten` step
  between the conv stack and the dense classifier *shows* where the
  architecture switches from "operating on spatial structure" to
  "operating on flattened feature vectors." In PyTorch that line is
  `x = x.view(N, -1)` and the change is invisible. In our code the
  type changes — `Tensor4D` becomes `Matrix` — and the reader cannot
  miss that something architectural has happened.


## Why production libraries chose Path A

The libraries that ML engineers use every day chose Path A, and they
are not wrong to. The reasons map almost exactly to where their goals
differ from ours:

- They are libraries for working practitioners, not learners. The
  type system isn't being asked to teach; it is being asked to stay
  out of the way.
- They serve every possible rank — 1-D vectors, 3-D sequences, 5-D
  video, dynamically-shaped data — and cannot afford one class per
  rank.
- They optimise broadcasting and reshape heavily, so a `.view()` call
  is essentially free and idiomatic.
- They have automatic differentiation, which records computational
  graphs over generic tensors and would be enormously complicated by
  per-rank types.

For those goals, unified is the correct choice. For ours — teaching
— separate is correct. Same problem, different priorities, different
answers.


## The principle, and a look ahead

The principle to carry forward is this: **each data structure earns
its own type when it has its own conceptual identity.** Matrix has an
identity — it is the object on which the backprop math is written.
Tensor4D has an identity — it is what a batch of feature maps looks
like, and the operations on it (convolution, pooling) are different
in character from matrix operations. If a third structure came along
that was *only* "matrix with a bigger shape" we would not necessarily
need a new class for it.

When attention arrives later in the book, the data shape will be
3-D — batch × time × features. The question will repeat: a new
`Tensor3D` class, or a reshape of `Matrix`? The honest answer will
depend on whether the operations are meaningfully new. If they are —
and attention has its own characteristic operations (scaled dot
product, masking, positional encoding) — a new type may earn its
keep. If they are not, reuse.

If this library ever grew into a more general framework — N-D tensors,
broadcasting, autograd, GPU support — refactoring toward a unified
`Tensor` would be a natural step. The current design is not a mistake
to be undone; it is a deliberate choice for an educational stage, and
a more general library would eventually outgrow it. That is a fair
thing to say in a book: the design *is* a design, with reasons.

What comes next is the layer that finally uses the Tensor4D — `Conv2D`,
which turns a `Tensor4D(N, in_C, H, W)` into a
`Tensor4D(N, out_C, H', W')` via the im2col-and-matmul approach. That
is the next note, and where the shape we just chose pays off in code.
