# 3 — Backpropagation, and the Dense layer

This chapter covers how a neural network learns. It builds on the
matrix class from the last chapter. We derive backpropagation for a
`Dense` layer, implement it in `src/Layer.h` and `src/Dense.h`, and
verify the implementation with a numerical gradient check.

---

## 1. Backprop is the chain rule, organized

A neural network is a composition of functions:

$$
L=\operatorname{Loss}\!\left(f_N(\cdots f_2(f_1(X))\cdots),
\mathrm{target}\right).
$$

`L` is a single scalar, the loss. Training means finding parameters
that minimise `L`. Gradient descent does this with the update

$$
p\leftarrow p-\eta\frac{\partial L}{\partial p}.
$$

So we need `∂L/∂p` for each weight and bias. The chain rule is the only
mathematical tool required. Backpropagation is the chain rule applied
bottom-up over the composition, organised so that intermediate
quantities are reused. It involves no new calculus, only careful
bookkeeping.

---

## 2. Each layer computes its gradients locally

This abstraction makes the problem tractable. A layer has a forward
role and a backward role.

**Forward.** It takes an input `X` and produces an output `Y = f(X)`.

**Backward.** It receives the *upstream gradient* `∂L/∂Y`, which says
how much the loss changes if the layer's output changes. From that it
computes:

- The *parameter gradients* `∂L/∂W`, `∂L/∂b`, ..., stored internally
  for the optimiser to apply later.
- The *downstream gradient* `∂L/∂X`, which is returned and becomes the
  upstream gradient of the layer below.

The computation is local. Each layer only needs to know how its own
forward computation depends on its own inputs and parameters. It knows
nothing about the layers above or below it. The global gradient is
assembled by composition, the same composition that defined the
network, run in reverse.

This is also why a network can be assembled layer by layer
(`net.add<Dense>(…)`, [chapter 8](08_network.md)): layers are
self-contained units that can be chained arbitrarily.

---

## 3. The intuition: perturbation propagation

Before the formulas, here is the idea for a Dense layer with
`Z = W·X + b`.

Pick a single weight `W[i,j]` and change it by a small `ε`. The forward
formula is

$$
Z_i=\sum_k W_{i,k}X_k+b_i.
$$

so only the `k = j` term changes. `Z[i]` changes by `ε · X[j]`, and no
other `Z[r ≠ i]` changes at all. That `ε · X[j]` propagates forward
through whatever comes next, and changes the loss by
`ε · X[j] · ∂L/∂Z[i]`. Dividing both sides by `ε`:

$$
\frac{\partial L}{\partial W_{i,j}}
=X_j\frac{\partial L}{\partial Z_i}.
$$

In words:

> A parameter's gradient is "how much it influences the output" times
> "how much the output influences the loss."

The first factor comes from the layer's own structure (its forward
formula). The second factor arrives from upstream. Backprop combines
each layer's local computation of its influence on the output with the
upstream measure of how that output affects the loss.

---

## 4. Deriving Dense's backward pass

The element-wise result above becomes a matrix formula:
`X[j] · ∂L/∂Z[i]` is the `(i,j)` entry of the outer product
`(∂L/∂Z) · X^T`. For a batch, every example contributes its own outer
product and they sum together. Writing the expression as a matmul
performs that summation. Letting `dZ` denote `∂L/∂Z`:

$$
dW=dZ\,X^\mathsf{T}.
$$

The shapes are `(out × batch)(batch × in) = (out × in)`.

**For the bias.** `Z[i,k] = ... + b[i]` for every example `k` in the
batch, so `∂Z[i,k] / ∂b[i] = 1`. Summing the chain rule across the
batch:

$$
db_i=\sum_{n=1}^{N}dZ_{i,n}.
$$

The result has shape `(out × 1)`.

This is what `Matrix::sumColumns()` returns, and it is the reason we
built that method.

**For the input gradient (passed down to the layer below).**
`Z[i,k] = Σ_j W[i,j] · X[j,k]`, so `∂Z[i,k] / ∂X[j,k] = W[i,j]`.
Summing over `i`:

$$
\frac{\partial L}{\partial X_{j,k}}
=\sum_i W_{i,j}\,dZ_{i,k},\qquad
dX=W^\mathsf{T}dZ.
$$

The matrix shapes are `(in × out)(out × batch) = (in × batch)`.

The full backward pass of Dense is:

```cpp
dW = dZ * X_cache.transpose();   // (out × in)
db = dZ.sumColumns();            // (out × 1)
dX = W.transpose() * dZ;         // (in × batch), passed down
```

---

## 5. The transpose symmetry and a shape check

Forward sends the signal through `W`, and backward sends the gradient
through `W^T`: the same matrix, transposed. This follows from a
duality. A matrix `W` of shape `(m × n)` maps an `n`-vector to an
`m`-vector going forward, so its dual mapping `m → n` going backward
has to be the transpose.

The practical consequence: whenever you are unsure about a backprop
formula, check the shapes. The dimensions tell you whether you need a
transpose, which side of the product something belongs on, and so on.
We have found many gradient bugs this way.

---

## 6. Implementation notes

### Caching

`forward()` saves a copy of `X` in `X_cache_`. `backward()` needs it to
compute `dW = dZ · X^T`. The cache belongs to the most recent forward
call. Run its backward pass before another forward overwrites that
input, and update the parameters after backward has used the original
weights.

### The Layer base class

The core of the interface (the real class also carries a training flag,
a `describe()` string and parameter save/load, which later chapters
introduce):

```cpp
template <typename T = float>
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix<T> forward(const Matrix<T>& X)   = 0;
    virtual Matrix<T> backward(const Matrix<T>& dY) = 0;
    virtual void update(Optimizer<T>& /*opt*/) {}   // no-op default
};
```

`update` has a default no-op so activation layers (ReLU, Softmax) that
have no parameters don't need to implement anything. `Dense::update`
overrides it to hand each parameter to the optimiser; the optimiser
decides how to apply the update.

### Initialisation

We initialise `W` from a Gaussian with stddev `√(2 / fan_in)`, which is
**He initialisation**. It keeps the variance of activations roughly
constant across layers (so signals neither shrink to zero nor grow
without bound), and it is the right default when paired with ReLU
[@he2015delving].

### Where the `1/batch_size` goes

When a per-example loss is averaged over the batch, an extra `1/N`
factor has to go somewhere. We adopt the convention that the loss
function includes the `1/N` in `dL/dZ`, so the layer backward formulas
stay simple (no batch-size arithmetic inside Dense). The loss class,
added later, shows this directly.

---

## 7. Testing with the numerical gradient check

A sign error, missing transpose, or forgotten sum across the batch can
leave plausible-looking outputs while giving the wrong update.
A **numerical gradient check** compares backward with the change in the
implemented forward calculation.

For every parameter `p` independently, we compare:

- **Analytical**: the gradient that `backward()` computed.
- **Numerical**: the definition of a derivative,
  `(L(p+ε) − L(p−ε)) / (2ε)`.

Agreement checks the derivative of that implementation at the sampled
points. It does not establish that the forward function is the intended
objective or that every input regime is covered. (Our `float` tests use
`ε = 10⁻³` and accept a difference below
`10⁻²`; later tests in `double` demand agreement to `10⁻⁸` or better.)
If they disagree, first check the finite-difference step and whether the
perturbation crosses a nondifferentiable point, such as ReLU at zero.
A persistent discrepancy at smooth points is evidence of a derivative
error.

In `tests/test_dense.cpp`:

1. **Forward** is verified against a tiny hand-computed example.
2. **Backward** is verified against the same example by hand.
3. **Numerical vs analytical** is run on a `(3 in, 4 out, batch 5)`
   layer for every entry of `dW` and `db`. (We use the trivial loss
   `L = sum(Z)` so `dL/dZ` is a matrix of ones, the simplest possible
   setup.)
4. **End-to-end SGD** trains a Dense layer for 50 steps on a quadratic
   loss and confirms the loss drops sharply, which shows that the
   forward/backward/update loop is connected correctly.

The two-sided difference `(L(p+ε) − L(p−ε)) / (2ε)` is preferred over
the one-sided `(L(p+ε) − L(p)) / ε`: it has error `O(ε²)` instead of
`O(ε)`, so you can use a larger `ε` and get a more accurate estimate.
For a stricter check, use `double` and a tight tolerance.

---

## 8. The following chapters {#sec-03-backprop-the-following-chapters}

A `Dense` layer alone can only fit linear maps. To make a network
expressive we need **nonlinearities**: `ReLU`, then `Softmax` paired
with **cross-entropy loss**. [Chapter 4](04_one_weight.md) first redoes
this chapter's arithmetic by hand on a single weight.
[Chapter 5](05_activations.md) adds `ReLU` and `Softmax`,
[chapter 7](07_cross_entropy.md) the cross-entropy loss and
[chapter 8](08_network.md) the `Network` class. With these we have
everything for our first classifier (IRIS, [chapter 9](09_iris.md)).


## Further reading {#sec-03-backprop-further-reading}

Backpropagation as a practical way to train multi-layer networks was
established by [@rumelhart1986backprop]. The initialisation that `Dense`
uses, a Gaussian with variance `2 / fan_in`, comes from [@he2015delving],
which extends the earlier analysis of [@glorot2010understanding] to ReLU
networks.
