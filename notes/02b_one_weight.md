# 02b — One weight, one step (SGD by hand)

The main backprop note (`02_backprop.md`) derives the matrix formulas for
`Dense::backward`. This note walks through the *same arithmetic* on a single
neuron with a single weight, by hand, with concrete numbers. The point is to
see exactly what the matrix formulas are mass-producing.

The runnable companion is `backprop_example/backprop_example.cpp` — it does
this calculation in C++ with both raw scalars and `weft::Dense` side by side
and prints the numbers.

---

## The setup

The smallest possible "network": a single neuron with two inputs and no
activation function (we'll add activations in the next session — the linear
part alone is enough to see the mechanism).

| symbol | value | meaning |
|--------|------:|---------|
| `x₁`   |   2.0 | first input |
| `x₂`   |  -1.0 | second input |
| `w₁`   |   0.5 | weight on `x₁` |
| `w₂`   |   1.0 | weight on `x₂` |
| `b`    |   0.1 | bias |
| `t`    |   1.0 | target output |
| `η`    |   0.1 | learning rate |

We'll mostly follow `w₁` and treat `w₂` and `b` as frozen, then loop back at
the end.

---

## Step 1 — Forward pass (the projection)

The neuron computes a weighted sum:

```
z = w₁·x₁ + w₂·x₂ + b
  = 0.5·2 + 1.0·(-1) + 0.1
  = 1 - 1 + 0.1
  = 0.1
```

The network output is `z = 0.1` but we wanted `t = 1.0`. The network is wrong
— off by 0.9 in the "z is too small" direction.

---

## Step 2 — Turning "off by 0.9" into a single number: the loss

"Off by 0.9" is a feeling, not a quantity we can differentiate. We pick a
loss function to turn it into a scalar. The simplest is squared error:

```
L = ½(z - t)²
  = ½(0.1 - 1.0)²
  = ½·(0.81)
  = 0.405
```

Three things worth noting about this choice:

- The square makes `L` positive regardless of which side of the target we
  land on.
- It also penalises being very wrong more than being slightly wrong.
- The `½` out front is cosmetic — it cancels with the `2` from
  differentiation, leaving cleaner formulas. You can drop it and rescale
  `η`; nothing important changes.

---

## Step 3 — The "error signal" ∂L/∂z

Backprop starts by asking: *if `z` were nudged a tiny bit, how would `L`
respond?* That's `∂L/∂z`. For squared error:

```
∂L/∂z = z - t = 0.1 - 1.0 = -0.9
```

The *sign* tells us which way is downhill (negative → nudging `z` up would
reduce `L`). The *magnitude* (0.9) tells us how steeply `L` responds. This
single number is what's often called the "error signal" or "delta" at the
output. In a deeper network it's exactly what gets passed backward into the
previous layer.

---

## Step 4 — Backpropagating to w₁ via the chain rule

We want `∂L/∂w₁`. The chain rule:

```
∂L/∂w₁ = (∂L/∂z) · (∂z/∂w₁)
```

We already have `∂L/∂z = -0.9`. For `∂z/∂w₁`, look at the forward formula
`z = w₁·x₁ + w₂·x₂ + b`. If we nudge `w₁` by `ε`, only the first term moves
— by `ε·x₁`. So `∂z/∂w₁ = x₁ = 2`. Therefore:

```
∂L/∂w₁ = (-0.9) · 2 = -1.8
```

Read it in English: a unit increase in `w₁` would increase `z` by 2
(because `x₁ = 2`), and a unit increase in `z` would decrease `L` at a rate
of 0.9. Multiplying these together: pushing `w₁` up by 1 would (linearly)
decrease `L` by 1.8. The gradient is `-1.8`, so to make `L` smaller we
should make `w₁` larger.

> **A parameter's gradient = (how much it influences the output) × (how much
> the output influences the loss).**

That's the entire backprop machinery for one weight, in one line of
arithmetic.

---

## Step 5 — The SGD update

Plain SGD steps each weight against its gradient:

```
w₁ ← w₁ - η · ∂L/∂w₁
    = 0.5 - 0.1 · (-1.8)
    = 0.5 + 0.18
    = 0.68
```

The two minus signs flip — the gradient was negative, so the update is
positive. We push `w₁` up, which makes `z` bigger, which is what we want.

The learning rate `η` controls how big a step we take. Too small and
learning crawls; too big and we overshoot and `L` can actually go *up*.

---

## Step 6 — Did it actually help?

Forward pass again with the updated `w₁` (others still frozen):

```
new z = 0.68·2 + 1.0·(-1) + 0.1 = 0.46
new L = ½(0.46 - 1.0)² = 0.1458
```

`L` dropped from 0.405 to 0.146 — about a 3× improvement from moving one
weight. If we update `w₂` and `b` simultaneously (their gradients work out
to `+0.9` and `-0.9`, giving new values 0.91 and 0.19), the next forward
pass lands at `z = 0.64`, `L = 0.065`. Closer still.

This is the training algorithm. Run it on many (input, target) pairs with
many weights, and a network learns.

---

## Where is "squared error" in the SGD update?

This is the right question to ask, because the SGD update rule
`w ← w - η · ∂L/∂w` doesn't *mention* a loss function. So where is it?

The loss is baked into the gradient. Trace where it appeared:

- **Step 2** computed `L = 0.405` — but `L` itself was never used again. It
  was reported for our information, then discarded.
- **Step 3** computed `∂L/∂z = z - t`. **This is the only place the squared
  error lives.** That `z - t` is the derivative of `½(z - t)²`.
- **Step 4** combined that number with the chain rule. The chain rule has
  no idea which loss produced the `-0.9`; it just multiplies.
- **Step 5** stepped against the gradient. By this point the loss has done
  its job — we have a number, and SGD steps against it.

If we used absolute error `L = |z - t|` instead, only step 3 would change
(to `∂L/∂z = sign(z - t)`). Cross-entropy with softmax would change just
that one line too. Everything else — chain rule, bias gradient, weight
update — is identical.

This is why in the library we keep **losses** and **optimisers** as
separate, swappable concepts. A loss class produces the initial `∂L/∂Z`
given the prediction and target; an optimiser applies whatever gradients
arrive however it likes (SGD, momentum, Adam, …). The two don't need to
know about each other.

---

## How this scales up to the matrix formulas

Look at what `Dense::backward` actually does, for our one-neuron example:

**`dW = dZ * X^T`** — `dZ` is the 1×1 matrix `[-0.9]`, `X^T` is the row
`[2, -1]`. Their product is `[-1.8, 0.9]` — exactly the two chain-rule
products we did by hand (`-1.8` for `w₁`, `0.9` for `w₂`). The matmul is
mass-producing the single-weight calculation for every weight at once.

**`db = dZ.sumColumns()`** — for one example, this is just `dZ` itself:
`[-0.9]`. By hand: `∂L/∂b = ∂L/∂z · 1 = -0.9`. Same.

**`dX = W^T * dZ`** — `W = [0.5, 1.0]`, so `W^T · [-0.9] = [-0.45, -0.9]^T`.
This is `∂L/∂xⱼ = wⱼ · ∂L/∂z` — "if `xⱼ` had been a bit larger, the loss
would have changed at this rate." It's what the layer below needs to know
in order to update its own weights. We don't have a layer below here, so
we ignore it. Stack two `Dense` layers and `dX` from the second becomes
the `dZ` going into the first.

Every matrix line in `Dense::backward` is producing the per-weight
arithmetic from this note — one for every weight in `W`, summed across
every example in the batch. The math doesn't get fancier when you go from
one weight to a million; only the bookkeeping does, and that's what
matrices are for.

---

## See it run

`backprop_example/backprop_example.cpp` runs this exact calculation twice
— once with raw scalars (mirroring this note line by line), once with
`weft::Dense` — and prints both sets of numbers so you can verify they
match.

Build it from the project root with `make -C build` and run
`./build/bin/examples/backprop_example`.
