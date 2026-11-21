# 4 — One weight, one step (SGD by hand)

The main backprop chapter ([chapter 3](03_backprop.md)) derives the
matrix formulas for `Dense::backward`. This chapter goes through the
same arithmetic on a single neuron with a single weight, by hand, with
concrete numbers. It shows the calculation that the matrix formulas
repeat for every weight.

The runnable companion is `examples/backprop_example.cpp`. It does this
calculation in C++ with both raw scalars and `weft::Dense` side by side
and prints the numbers.

---

## The setup

The network is as small as possible: a single neuron with two inputs
and no activation function. Activations come in the next chapter; the
linear part alone is enough to show the mechanism.

| symbol | value | meaning |
|--------|------:|---------|
| `x₁`   |   2.0 | first input |
| `x₂`   |  -1.0 | second input |
| `w₁`   |   0.5 | weight on `x₁` |
| `w₂`   |   1.0 | weight on `x₂` |
| `b`    |   0.1 | bias |
| `t`    |   1.0 | target output |
| `η`    |   0.1 | learning rate |

We mostly follow `w₁` and treat `w₂` and `b` as frozen, then return to
them at the end.

---

## Step 1 — Forward pass (the projection)

The neuron computes a weighted sum:

```
z = w₁·x₁ + w₂·x₂ + b
  = 0.5·2 + 1.0·(-1) + 0.1
  = 1 - 1 + 0.1
  = 0.1
```

The network output is `z = 0.1` but we wanted `t = 1.0`. The output is
too small by 0.9.

---

## Step 2 — Turning "off by 0.9" into a single number: the loss

"Off by 0.9" is not yet a quantity we can differentiate. We pick a loss
function to turn it into a scalar. The simplest is squared error:

```
L = ½(z - t)²
  = ½(0.1 - 1.0)²
  = ½·(0.81)
  = 0.405
```

This choice has the following properties:

- The square makes `L` positive regardless of which side of the target
  we land on.
- It also penalises being very wrong more than being slightly wrong.
- The `½` in front is cosmetic. It cancels with the `2` from
  differentiation, which gives cleaner formulas. You can drop it and
  rescale `η`; nothing important changes.

---

## Step 3 — The "error signal" ∂L/∂z

Backprop starts by asking how `L` responds if `z` is changed by a small
amount. That is `∂L/∂z`. For squared error:

```
∂L/∂z = z - t = 0.1 - 1.0 = -0.9
```

The sign tells us the direction of decrease (negative means that
increasing `z` would reduce `L`). The magnitude (0.9) tells us how
steeply `L` responds. This number is often called the "error signal" or
"delta" at the output. In a deeper network it is what gets passed
backward into the previous layer.

---

## Step 4 — Backpropagating to w₁ via the chain rule

We want `∂L/∂w₁`. The chain rule gives:

```
∂L/∂w₁ = (∂L/∂z) · (∂z/∂w₁)
```

We already have `∂L/∂z = -0.9`. For `∂z/∂w₁`, look at the forward
formula `z = w₁·x₁ + w₂·x₂ + b`. If we change `w₁` by `ε`, only the
first term changes, by `ε·x₁`. So `∂z/∂w₁ = x₁ = 2`. Therefore:

```
∂L/∂w₁ = (-0.9) · 2 = -1.8
```

In words: a unit increase in `w₁` would increase `z` by 2 (because
`x₁ = 2`), and a unit increase in `z` would decrease `L` at a rate of
0.9. Multiplying these together, increasing `w₁` by 1 would (linearly)
decrease `L` by 1.8. The gradient is `-1.8`, so to make `L` smaller we
should make `w₁` larger.

> A parameter's gradient = (how much it influences the output) × (how
> much the output influences the loss).

For one weight, backprop is this one multiplication.

---

## Step 5 — The SGD update

Plain SGD moves each weight against its gradient:

```
w₁ ← w₁ - η · ∂L/∂w₁
    = 0.5 - 0.1 · (-1.8)
    = 0.5 + 0.18
    = 0.68
```

The two minus signs cancel: the gradient was negative, so the update is
positive. `w₁` increases, which increases `z`, as required.

The learning rate `η` controls the size of the step. If it is too
small, learning is slow. If it is too big, we overshoot and `L` can go
up.

---

## Step 6 — Checking the result

Forward pass again with the updated `w₁` (others still frozen):

```
new z = 0.68·2 + 1.0·(-1) + 0.1 = 0.46
new L = ½(0.46 - 1.0)² = 0.1458
```

`L` dropped from 0.405 to 0.146, about a 3× improvement from moving one
weight. If we update `w₂` and `b` at the same time (their gradients
work out to `+0.9` and `-0.9`, giving new values 0.91 and 0.19), the
next forward pass gives `z = 0.64`, `L = 0.065`, which is closer still.

This is the training algorithm. Run on many (input, target) pairs with
many weights, it trains a network.

![The loss as a function of `w₁`, with the other weight and the bias held fixed. The dashed line has the slope −1.8 computed in step 4. The update moves `w₁` from 0.5 to 0.68 and the loss from 0.405 to 0.1458.](figures/04_one_step.png)

---

## Where the loss enters the SGD update

The SGD update rule `w ← w - η · ∂L/∂w` does not mention a loss
function. The loss enters through the gradient:

- **Step 2** computed `L = 0.405`, but `L` itself was never used again.
  It was reported for our information, then discarded.
- **Step 3** computed `∂L/∂z = z - t`. This is the only place where the
  squared error appears: `z - t` is the derivative of `½(z - t)²`.
- **Step 4** combined that number with the chain rule. The chain rule
  does not depend on which loss produced the `-0.9`; it only
  multiplies.
- **Step 5** stepped against the gradient. By this point the loss is no
  longer needed: we have a number, and SGD steps against it.

If we used absolute error `L = |z - t|` instead, only step 3 would
change (to `∂L/∂z = sign(z - t)`). Cross-entropy with softmax would
also change only that line. Everything else (chain rule, bias gradient,
weight update) is identical.

This is why the library keeps **losses** and **optimisers** as
separate, swappable concepts. A loss class produces the initial `∂L/∂Z`
given the prediction and target. An optimiser applies whatever
gradients arrive, by its own rule (SGD, momentum, Adam, …). Neither
needs to know about the other.

---

## How this scales up to the matrix formulas

Here is what `Dense::backward` does for our one-neuron example.

**`dW = dZ * X^T`**. `dZ` is the 1×1 matrix `[-0.9]`, `X^T` is the row
`[2, -1]`. Their product is `[-1.8, 0.9]`, the two chain-rule products
we did by hand (`-1.8` for `w₁`, `0.9` for `w₂`). The matmul performs
the single-weight calculation for every weight at once.

**`db = dZ.sumColumns()`**. For one example this is `dZ` itself:
`[-0.9]`. By hand: `∂L/∂b = ∂L/∂z · 1 = -0.9`. The values agree.

**`dX = W^T * dZ`**. `W = [0.5, 1.0]`, so
`W^T · [-0.9] = [-0.45, -0.9]^T`. This is `∂L/∂xⱼ = wⱼ · ∂L/∂z`, the
rate at which the loss would have changed if `xⱼ` had been a little
larger. The layer below needs it in order to update its own weights.
There is no layer below here, so we ignore it. Stack two `Dense` layers
and `dX` from the second becomes the `dZ` going into the first.

Every matrix line in `Dense::backward` produces the per-weight
arithmetic from this chapter, once for every weight in `W`, summed
across every example in the batch. Going from one weight to a million
does not change the math. It only adds bookkeeping, which the matrices
handle.

---

## Running the example

`examples/backprop_example.cpp` runs this calculation twice: once with
raw scalars (mirroring this chapter, except that it updates all three
parameters at once, so its last line is the `z = 0.64`, `L = 0.065`
case) and once with `weft::Dense`. It prints both sets of numbers so
you can verify that they match.

Build it with `make` in the `build/` directory and run
`./examples/backprop_example` from there.


## Further reading

Updating parameters from one noisy gradient estimate at a time goes back
to the stochastic approximation method of [@robbins1951sgd].
