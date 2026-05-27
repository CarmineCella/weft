# 03b — The Jacobian (the unifying object behind every backward pass)

`02_backprop.md` derived `Dense::backward` from a perturbation argument on a
single weight. `03_activations.md` built ReLU and Softmax and derived their
backward formulas. This note steps back and shows that *every* backward pass
we've written — and every one we'll write — is the same operation:

> **Backward = multiply the upstream gradient by the transposed Jacobian of
> the layer's forward function.**

Layers differ only in what their Jacobian looks like and how cheaply we can
multiply by it.

---

## 1. What a Jacobian is

Build up from what's familiar.

### Scalar function of a scalar
For `f: ℝ → ℝ`, the derivative `f'(x)` is one number — the slope of `f` at
`x`. It says: if I nudge `x` by `ε`, `f` changes by approximately
`f'(x) · ε`.

### Scalar function of a vector
For `f: ℝⁿ → ℝ`, we need one partial per input, packaged as a vector — the
**gradient**:

```
∇f = [ ∂f/∂x₁,  ∂f/∂x₂,  …,  ∂f/∂xₙ ]
```

Same content as the slope, but vector-valued because there are multiple
directions you can nudge.

### Vector function of a vector
For `f: ℝⁿ → ℝᵐ` — `m` outputs, `n` inputs — each output has its own
gradient. Stack them as rows and you get a matrix: the **Jacobian**.

```
       ┌                              ┐
       │ ∂f₁/∂x₁  ∂f₁/∂x₂  …  ∂f₁/∂xₙ │      shape: m × n
   J = │ ∂f₂/∂x₁  ∂f₂/∂x₂  …  ∂f₂/∂xₙ │
       │   ⋮         ⋮              ⋮ │
       │ ∂fₘ/∂x₁  ∂fₘ/∂x₂  …  ∂fₘ/∂xₙ │
       └                              ┘
```

`J[i,j] = ∂fᵢ/∂xⱼ` — "how does output `i` respond to a nudge in input `j`?"

Row `i` is "the gradient of output `i`." Column `j` is "the effect of input
`j` on every output."

### The geometric picture

The Jacobian is the **local linearisation** of `f` at the point `x`. For
small `ε`:

```
f(x + ε)  ≈  f(x)  +  J · ε
```

This is the multi-variable generalisation of `f(x + ε) ≈ f(x) + f'(x)·ε`.
It's exactly the perturbation intuition we used to derive Dense's backward
pass in `02b_one_weight.md` — but stated for vectors. The Jacobian answers
"how does the output respond when an input is nudged?" all at once, for
every (input, output) pair.

---

## 2. Backprop is "multiply by the transposed Jacobian"

The chain rule for a layer `f: x → y` reads:

```
∂L/∂xⱼ  =  Σᵢ (∂L/∂yᵢ) · (∂yᵢ/∂xⱼ)
        =  Σᵢ (dL/dyᵢ) · J[i,j]
```

That sum is exactly the `j`-th entry of `Jᵀ · dL/dy`. So in matrix form:

```
dL/dx = Jᵀ · dL/dy
```

That single formula is the entire content of "backward" for any layer.
Backprop is the chain rule; the chain rule for vector functions is matrix
multiplication by the transposed Jacobian. *Every* `backward()` we will ever
write — Dense, ReLU, Softmax, Conv, BatchNorm, attention — is a
specialisation of this one expression.

What changes from layer to layer is (1) what the Jacobian looks like and
(2) how cheaply we can multiply by it *without* materialising it.

---

## 3. The Jacobians of weft's three layers

This view lets you re-read everything we've built as variations on
`Jᵀ · dL/dy`.

### Dense: J *is* W

`y = Wx + b`, so `∂yᵢ/∂xⱼ = W[i,j]`. **The Jacobian is `W` itself.** No
materialisation needed — `W` is already sitting in memory as a layer
parameter. The chain rule gives:

```
dX = Wᵀ · dY     i.e.    dX = Jᵀ · dY
```

The `Wᵀ` that appears in `Dense::backward` is just `Jᵀ` for this particular
Jacobian. The "transpose symmetry" of `02_backprop.md` is the general `Jᵀ`
rule, specialised to the linear case.

### ReLU: J is diagonal

`y = max(0, z)` element-wise, so `∂yᵢ/∂zⱼ = 0` whenever `i ≠ j`. The
Jacobian is **diagonal**, with:

```
J[i,i] = 1 if zᵢ > 0 else 0
```

Multiplying by a diagonal matrix is the same as element-wise multiplying by
its diagonal — a Hadamard product. That is why `dZ = dA ⊙ mask` is correct,
and why we store only the `K`-element mask, never the `K × K` matrix.

### Softmax: J is dense, but structured

`sᵢ = exp(zᵢ) / Σₖ exp(zₖ)`. Every output depends on every input, so:

```
J[i,j] = ∂sᵢ/∂zⱼ = sᵢ · (δᵢⱼ - sⱼ)
```

This is the first non-diagonal Jacobian in the library. Two structural
properties worth noticing:

- **It's symmetric** (`J = Jᵀ`): the off-diagonal `-sᵢ · sⱼ` is symmetric in
  `i` and `j`, and the diagonal entries `sᵢ(1 - sᵢ)` are well-defined. So
  `Jᵀ · dS = J · dS` — convenient.
- **Each row sums to zero**: because the outputs `sᵢ` always sum to 1,
  nudging any `zⱼ` rearranges the `sᵢ` without changing their total. So
  `Σᵢ ∂sᵢ/∂zⱼ = 0` for any `j`. Useful sanity check when debugging.

---

## 4. A concrete worked example

Pick `z = (1, 0, -1)`. After the stability shift (subtract `max = 1`):

```
e_z'   = (e⁰, e⁻¹, e⁻²)  ≈  (1, 0.368, 0.135)
D      = 1 + 0.368 + 0.135  ≈  1.503
s      ≈ (0.665, 0.245, 0.090)
```

Build `J` from `J[i,j] = sᵢ(δᵢⱼ - sⱼ)`:

```
       ┌                          ┐
   J ≈ │  0.223  -0.163  -0.060   │
       │ -0.163   0.185  -0.022   │
       │ -0.060  -0.022   0.082   │
       └                          ┘
```

(Symmetric ✓, every row sums to ~0 ✓.)

Pick an arbitrary upstream gradient `dS = (0.5, -0.3, 0.1)` and compute the
downstream `dZ = J · dS` two ways.

**Method 1 — materialise `J` and matmul:**

```
dZ[0] =  0.223·0.5 + (-0.163)·(-0.3) + (-0.060)·0.1  ≈  0.154
dZ[1] = -0.163·0.5 +   0.185·(-0.3)  + (-0.022)·0.1  ≈ -0.139
dZ[2] = -0.060·0.5 + (-0.022)·(-0.3) +   0.082·0.1   ≈ -0.015
```

**Method 2 — the algebraic shortcut implemented in `Softmax::backward`:**

```
dot = s · dS  =  0.665·0.5 + 0.245·(-0.3) + 0.090·0.1  ≈  0.268

dZ[0] = s₀(dS₀ - dot) = 0.665·(0.5 - 0.268)   ≈  0.154
dZ[1] = s₁(dS₁ - dot) = 0.245·(-0.3 - 0.268)  ≈ -0.139
dZ[2] = s₂(dS₂ - dot) = 0.090·(0.1 - 0.268)   ≈ -0.015
```

Identical results. The shortcut is precisely the algebra of `J · dS` with
the structure of `J` factored out — so we never have to build `J`.

---

## 5. Never materialise a Jacobian you can avoid

For `K` classes, Softmax's explicit Jacobian is a `K × K` matrix *per
example*. With `K = 1000` and a batch of 32, that's 32 million floats just
to hold the Jacobians — plus a 1000×1000 matrix-vector product per
example.

The shortcut formula avoids both: one dot product (`O(K)`) plus one
element-wise pass (`O(K)`) per column. **`O(K)` instead of `O(K²)`, with
zero extra allocation.**

This is the general principle of efficient backprop:

> **If your Jacobian has structure, exploit it. Never build a Jacobian you
> can avoid building.**

Every layer in this library follows the rule:

- **Dense**: Jacobian *is* `W` — a parameter we already have.
- **ReLU**: Jacobian is diagonal — we store only the diagonal, as a mask.
- **Softmax**: Jacobian has rank-1 correction structure — one dot product
  per column captures the action.
- **Convolution** (later): the Jacobian is enormous but extremely sparse and
  shift-invariant — the action becomes another (transposed) convolution.

The skill of writing efficient layers is recognising the structure of `J`
and reducing `Jᵀ · dY` to the cheapest equivalent operation.

---

## 6. Summary

The Jacobian is the single object underneath every backward pass — the
matrix of all first-order responses of a layer's outputs to its inputs. The
chain rule for vector functions is matrix multiplication by the transposed
Jacobian:

```
dL/dx = Jᵀ · dL/dy
```

That formula, plus the principle "exploit structure, never materialise an
expensive Jacobian", is the whole story of backward in a neural-network
library. Everything else is recognising what `J` looks like for your
particular layer and rewriting `Jᵀ · dY` in the cheapest possible form.
