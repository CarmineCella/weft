# 6 — The Jacobian (the unifying object behind every backward pass)

[Chapter 3](03_backprop.md) derived `Dense::backward` from a perturbation
argument on a single weight. [Chapter 5](05_activations.md) built ReLU and
Softmax and derived their backward formulas. This chapter shows that
every backward pass we have written, and every one we will write, is
the same operation:

> Backward = multiply the upstream gradient by the transposed Jacobian of
> the layer's forward function.

Layers differ only in what their Jacobian looks like and how cheaply we can
multiply by it.

---

## 1. What a Jacobian is

We build up from the scalar case.

### Scalar function of a scalar
For `f: ℝ → ℝ`, the derivative `f'(x)` is one number, the slope of `f`
at `x`. If `x` changes by `ε`, `f` changes by approximately
`f'(x) · ε`.

### Scalar function of a vector
For `f: ℝⁿ → ℝ`, we need one partial derivative per input, collected in
a vector, the **gradient**:

$$
\nabla f=
\begin{bmatrix}
\partial f/\partial x_1 & \cdots & \partial f/\partial x_n
\end{bmatrix}^{\mathsf{T}}.
$$

It has the same content as the slope, but is vector-valued because
there are several directions in which the input can change.

### Vector function of a vector
For `f: ℝⁿ → ℝᵐ`, with `m` outputs and `n` inputs, each output has its
own gradient. Stacked as rows they form a matrix, the **Jacobian**.

$$
J=
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots &
\dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots &
\dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}
\in\mathbb{R}^{m\times n}.
$$

`J[i,j] = ∂fᵢ/∂xⱼ` says how output `i` responds to a small change in
input `j`.

Row `i` is the gradient of output `i`. Column `j` is the effect of
input `j` on every output.

### The geometric picture

The Jacobian is the **local linearisation** of `f` at the point `x`. For
small `ε`:

$$
f(x+\varepsilon)\approx f(x)+J\varepsilon.
$$

This is the multi-variable generalisation of `f(x + ε) ≈ f(x) + f'(x)·ε`.
It is the perturbation argument we used to derive Dense's backward pass
in [chapter 3](03_backprop.md), stated for vectors. The Jacobian gives
the response of every output to a small change in every input.

---

## 2. Backprop is "multiply by the transposed Jacobian"

The chain rule for a layer `f: x → y` reads:

$$
\frac{\partial L}{\partial x_j}
=\sum_i\frac{\partial L}{\partial y_i}
       \frac{\partial y_i}{\partial x_j}
=\sum_i\frac{\partial L}{\partial y_i}J_{i,j}.
$$

That sum is the `j`-th entry of `Jᵀ · dL/dy`. In matrix form:

$$
\nabla_x L=J^\mathsf{T}\nabla_y L.
$$

This formula is the content of "backward" for any layer. Backprop is
the chain rule, and the chain rule for vector functions is matrix
multiplication by the transposed Jacobian. Every `backward()` we write
(Dense, ReLU, Softmax, Conv, LayerNorm, attention) is a special case of
this expression.

What changes from layer to layer is (1) what the Jacobian looks like and
(2) how cheaply we can multiply by it without materialising it.

---

## 3. The Jacobians of weft's three layers

With this view, everything we have built is a variation on
`Jᵀ · dL/dy`.

### Dense: J is W

`y = Wx + b`, so `∂yᵢ/∂xⱼ = W[i,j]`. The Jacobian is `W` itself.
Nothing needs to be materialised, because `W` is already in memory as a
layer parameter. The chain rule gives:

$$
dX=W^\mathsf{T}dY=J^\mathsf{T}dY.
$$

The `Wᵀ` that appears in `Dense::backward` is `Jᵀ` for this particular
Jacobian. The "transpose symmetry" of [chapter 3](03_backprop.md) is the
general `Jᵀ` rule, specialised to the linear case.

### ReLU: J is diagonal

`y = max(0, z)` element-wise, so `∂yᵢ/∂zⱼ = 0` whenever `i ≠ j`. The
Jacobian is **diagonal**, with:

$$
J_{i,i}=
\begin{cases}
1,&z_i>0,\\
0,&z_i<0.
\end{cases}
$$

At zero the derivative is undefined; the implementation chooses zero
for its backward rule.

Multiplying by a diagonal matrix is the same as multiplying
element-wise by its diagonal, a Hadamard product. That is why
`dZ = dA ⊙ mask` is correct, and why we store only the `K`-element
mask, never the `K × K` matrix.

### Softmax: J is dense, but structured

`sᵢ = exp(zᵢ) / Σₖ exp(zₖ)`. Every output depends on every input, so:

$$
J_{i,j}=\frac{\partial s_i}{\partial z_j}
=s_i(\delta_{ij}-s_j).
$$

This is the first non-diagonal Jacobian in the library. It has
structure:

- **It is symmetric** (`J = Jᵀ`): the off-diagonal `-sᵢ · sⱼ` is
  symmetric in `i` and `j`, and the diagonal entries `sᵢ(1 - sᵢ)` are
  well-defined. So `Jᵀ · dS = J · dS`.
- **Each column sums to zero**: because the outputs `sᵢ` always sum to 1,
  changing any `zⱼ` rearranges the `sᵢ` without changing their total. So
  `Σᵢ ∂sᵢ/∂zⱼ = 0` for any `j`. Since `J` is symmetric, each row sums to
  zero too. This is a useful check when debugging.

---

## 4. A worked example

Take `z = (1, 0, -1)`. After the stability shift (subtract `max = 1`):

```
e_z'   = (e⁰, e⁻¹, e⁻²)  ≈  (1, 0.368, 0.135)
D      = 1 + 0.368 + 0.135  ≈  1.503
s      ≈ (0.665, 0.245, 0.090)
```

Build `J` from `J[i,j] = sᵢ(δᵢⱼ - sⱼ)`:

$$
J\approx
\begin{bmatrix}
 0.223 & -0.163 & -0.060\\
-0.163 &  0.185 & -0.022\\
-0.060 & -0.022 &  0.082
\end{bmatrix}.
$$

(The matrix is symmetric and every row sums to approximately 0.)

Take an arbitrary upstream gradient `dS = (0.5, -0.3, 0.1)` and compute
the downstream `dZ = J · dS` in two ways.

**Method 1: materialise `J` and matmul.**

```
dZ[0] =  0.223·0.5 + (-0.163)·(-0.3) + (-0.060)·0.1  ≈  0.154
dZ[1] = -0.163·0.5 +   0.185·(-0.3)  + (-0.022)·0.1  ≈ -0.139
dZ[2] = -0.060·0.5 + (-0.022)·(-0.3) +   0.082·0.1   ≈ -0.015
```

**Method 2: the algebraic shortcut implemented in `Softmax::backward`.**

```
dot = s · dS  =  0.665·0.5 + 0.245·(-0.3) + 0.090·0.1  ≈  0.268

dZ[0] = s₀(dS₀ - dot) = 0.665·(0.5 - 0.268)   ≈  0.154
dZ[1] = s₁(dS₁ - dot) = 0.245·(-0.3 - 0.268)  ≈ -0.139
dZ[2] = s₂(dS₂ - dot) = 0.090·(0.1 - 0.268)   ≈ -0.015
```

The results are identical. The shortcut is the algebra of `J · dS` with
the structure of `J` factored out, so we never have to build `J`.

---

## 5. Never materialise a Jacobian you can avoid

For `K` classes, Softmax's explicit Jacobian is a `K × K` matrix per
example. With `K = 1000` and a batch of 32, that is 32 million floats
to hold the Jacobians, plus a 1000×1000 matrix-vector product per
example.

The shortcut formula avoids both: one dot product (`O(K)`) plus one
element-wise pass (`O(K)`) per column. The cost is `O(K)` instead of
`O(K²)`, without allocating a Jacobian.

This is the general principle of efficient backprop:

> If your Jacobian has structure, exploit it. Never build a Jacobian you
> can avoid building.

Every layer in this library follows the rule:

- **Dense**: the Jacobian is `W`, a parameter we already have.
- **ReLU**: the Jacobian is diagonal, and we store only the diagonal,
  as a mask.
- **Softmax**: the Jacobian has rank-1 correction structure, and one
  dot product per column captures its action.
- **Convolution** (later): the Jacobian is enormous but extremely
  sparse and shift-invariant, and its action becomes another
  (transposed) convolution.

Writing an efficient layer means recognising the structure of `J` and
reducing `Jᵀ · dY` to the cheapest equivalent operation.

---

## 6. Summary

The Jacobian is the matrix of all first-order responses of a layer's
outputs to its inputs. The chain rule for vector functions is matrix
multiplication by the transposed Jacobian:

```
dL/dx = Jᵀ · dL/dy
```

Every backward pass in the library is this formula, with `Jᵀ · dY`
rewritten in the cheapest form that the structure of `J` allows, so
that an expensive Jacobian is never materialised.


## Further reading {#sec-06-jacobian-further-reading}

The view of backpropagation as a sequence of vector-Jacobian products is
that of reverse-mode automatic differentiation; [@baydin2018autodiff] is
a thorough survey. Chapter 6 of [@goodfellow2016deep] derives the same
formulas in textbook form.
