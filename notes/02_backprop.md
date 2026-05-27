# 02 — Backpropagation, and the Dense layer

The matrix from the last note is the foundation. This note is the heart of the
library: how a neural network *learns*. We derive backpropagation for a `Dense`
layer, implement it in `src/Layer.h` and `src/Dense.h`, and verify the
implementation with a *numerical gradient check* — the single most important
debugging technique in deep learning.

---

## 1. Backprop is the chain rule, organized

A neural network is a composition of functions:

```
L  =  Loss( f_N( ... f_2( f_1( X ) ) ... ), target )
```

`L` is a single scalar — the loss. Training means finding parameters that
minimise `L`. Gradient descent does this with the update

```
p  <-  p  -  η · ∂L/∂p          for every parameter p
```

So all we need is `∂L/∂p` for each weight and bias. The chain rule is the only
mathematical tool we need; backpropagation is just *the chain rule applied
bottom-up over the composition, organised so we reuse intermediate
quantities*. There is no new calculus — only careful bookkeeping.

---

## 2. Each layer is a self-contained gradient router

This is the abstraction that makes the whole thing tractable. A layer plays two
roles:

**Forward.** It takes an input `X` and produces an output `Y = f(X)`.

**Backward.** It receives the *upstream gradient* `∂L/∂Y` (i.e. "how much does
the loss change if my output changes?"). From that it computes two things:

- The *parameter gradients* `∂L/∂W`, `∂L/∂b`, ... — stored internally for the
  optimiser to apply later.
- The *downstream gradient* `∂L/∂X` — returned, to become the upstream
  gradient of the layer below.

The crucial word is *local*. Each layer only needs to know how its own forward
computation depends on its own inputs and parameters. It has no idea what's
above or below it. The global gradient assembly is just composition — the same
composition that defined the network in the first place, run in reverse.

This is also why we can specify architectures in a config file: layers are
self-contained units that can be chained arbitrarily.

---

## 3. The intuition: perturbation propagation

Skip the formulas for a second. Here is the gut-level picture for a Dense
layer with `Z = W·X + b`:

Pick a single weight `W[i,j]`. Nudge it by a tiny `ε`. The forward formula is

```
Z[i] = Σ_k W[i,k] · X[k]  +  b[i]
```

so only the `k = j` term moved. **`Z[i]` changes by `ε · X[j]`** — and no other
`Z[r ≠ i]` changes at all. That `ε · X[j]` ripples forward through whatever
comes next, and ultimately changes the loss by `ε · X[j] · ∂L/∂Z[i]`.
Dividing both sides by `ε`:

```
∂L/∂W[i,j]  =  X[j] · ∂L/∂Z[i]
```

Said in words:

> **A parameter's gradient is "how much it influences the output" times "how
> much the output influences the loss."**

The first factor comes from the layer's own structure (its forward formula).
The second factor arrives from upstream. Backprop is just gluing local
"how-much-I-influence-things" computations to the upstream "how-much-that-
affects-the-loss" coming down from above.

---

## 4. Deriving Dense's backward pass

The element-wise result above immediately becomes a matrix formula:
`X[j] · ∂L/∂Z[i]` is the `(i,j)` entry of the outer product
`(∂L/∂Z) · X^T`. For a batch, every example contributes its own outer
product and they *sum together* — and that summation happens automatically
when you write it as a matmul. Letting `dZ` denote `∂L/∂Z`:

```
dW  =  dZ · X^T          shapes: (out × batch)(batch × in)  =  (out × in)
```

**For the bias.** `Z[i,k] = ... + b[i]` for every example `k` in the batch, so
`∂Z[i,k] / ∂b[i] = 1`. Summing the chain rule across the batch:

```
db  =  dZ summed across columns      shape: (out × 1)
```

This is exactly what `Matrix::sumColumns()` returns — which is why we built it.

**For the input gradient (sent upstream).** `Z[i,k] = Σ_j W[i,j] · X[j,k]`,
so `∂Z[i,k] / ∂X[j,k] = W[i,j]`. Summing over `i`:

```
∂L/∂X[j,k]  =  Σ_i W[i,j] · dZ[i,k]  =  (W^T · dZ)[j,k]

dX  =  W^T · dZ          shapes: (in × out)(out × batch)  =  (in × batch)
```

So Dense's full backward pass is three lines of matrix algebra:

```cpp
dW = dZ * X_cache.transpose();   // (out × in)
db = dZ.sumColumns();            // (out × 1)
dX = W.transpose() * dZ;         // (in × batch), returned upstream
```

---

## 5. The transpose symmetry — and a free sanity check

Look at the pair: forward sends the signal through `W`, backward sends the
gradient through `W^T`. Same matrix, transposed. This isn't a coincidence —
it's a fundamental duality. A matrix `W` of shape `(m × n)` maps an
`n`-vector to an `m`-vector going forward, so its dual mapping `m → n` going
backward has to be the transpose.

The practical consequence: **whenever you're unsure about a backprop formula,
check the shapes**. The dimensions tell you whether you need a transpose,
which side of the product something belongs on, and so on. I've debugged more
gradients with this trick than I'd like to admit.

---

## 6. Implementation notes

### Caching

`forward()` saves a copy of `X` in `X_cache_`. `backward()` needs it to compute
`dW = dZ · X^T`. For a 32-example batch this copy is trivial; for larger
batches we can revisit, but it's almost never the bottleneck.

### The Layer base class

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

We initialise `W` from a Gaussian with stddev `√(2 / fan_in)` — **He
initialisation**. It keeps the variance of activations roughly constant across
layers (so signals don't shrink to zero or explode), and it's the right
default when paired with ReLU. We'll discuss why properly when ReLU arrives.

### Where the `1/batch_size` lives (decision worth flagging)

When you mean-average a per-example loss, an extra `1/N` factor needs to live
somewhere. We adopt the convention that **the loss function bakes the `1/N`
into `dL/dZ`**, so layer backward formulas stay clean (no batch-size
arithmetic inside Dense). When we add a loss class, you'll see this directly.

---

## 7. Testing — and why the numerical gradient check matters more than anything

Hard truth: backprop code that runs without crashing and produces "reasonable-
looking" outputs is *almost always wrong on the first try*. A subtle sign
error, a missing transpose, a forgotten sum across the batch — these don't
crash, they just make your network not learn well, and you waste days. The
cure is the **numerical gradient check**.

For every parameter `p` independently:

- **Analytical**: the gradient that `backward()` computed.
- **Numerical**: the actual definition of a derivative,
  `(L(p+ε) − L(p−ε)) / (2ε)`.

If they agree to ~4 decimal places, you have proof your backward matches the
math. If they don't, you have a bug. There is no substitute.

In `tests/test_dense.cpp`:

1. **Forward** is verified against a tiny hand-computed example.
2. **Backward** is verified against the same example by hand.
3. **Numerical vs analytical** is run on a `(3 in, 4 out, batch 5)` layer for
   every entry of `dW` and `db`. (We use the trivial loss `L = sum(Z)` so
   `dL/dZ` is just a matrix of ones — cleanest possible setup.)
4. **End-to-end SGD** trains a Dense layer for 50 steps on a quadratic loss
   and confirms the loss drops sharply — proves the whole forward/backward/
   update loop is wired correctly.

The two-sided difference `(L(p+ε) − L(p−ε)) / (2ε)` is preferred over the
one-sided `(L(p+ε) − L(p)) / ε`: it has error `O(ε²)` instead of `O(ε)`, so
you can use a larger `ε` and get a more accurate estimate. Use `double` and a
tight tolerance if you're paranoid.

---

## 8. What's next

A `Dense` layer alone can only fit linear maps. To make a network expressive
we need **nonlinearities** — `ReLU`, then `Softmax` paired with **cross-
entropy loss**. That's the next note, after which we'll have everything
needed for our first real classifier (IRIS).
