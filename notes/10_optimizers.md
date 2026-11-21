# 10 — Optimizers (SGD and Adam)

So far the only optimiser we have passed to `net.update(opt)` is plain
stochastic gradient descent. Adam and similar optimisers need
per-parameter state, which is why the update is in a separate
`Optimizer` object and not in the layer. This chapter covers that
separation, the ideas behind Adam, and why it usually performs better.

---

## 1. Why plain SGD struggles

Plain SGD updates `param ← param − lr · g`. It has structural
weaknesses.

**One learning rate for every parameter.** Gradient magnitudes can vary
by orders of magnitude across parameters of the same network: early
layers vs. late layers, weights vs. biases, parameters that interact with
sparse vs. dense features. A single learning rate has to be small enough
not to diverge on the largest gradients, so all the other parameters
move too slowly.

**No memory between steps.** Each step uses only the current minibatch's
gradient. A gradient that points the same way step after step is treated
like a noisy oscillation. SGD applies every gradient in full, even if it
is mostly variance.

Adam addresses each weakness with a different idea and combines them.

---

## 2. Momentum — accelerate in consistent directions

Track an exponential moving average of the gradient:

```
m_t  =  β₁ · m_{t-1}  +  (1 − β₁) · g_t      β₁ ≈ 0.9
```

`m` is the recent average gradient direction. Steps in consistent
directions accumulate; oscillating directions cancel and `m` stays small.

The update then uses `m_t` instead of `g_t`. The optimiser speeds up
where the gradient direction stays the same over many steps, and
individual minibatches have little effect where the gradient is noisy.

---

## 3. RMSProp — per-parameter step sizes

Track an exponential moving average of squared gradients, per
parameter:

```
v_t  =  β₂ · v_{t-1}  +  (1 − β₂) · g_t²     β₂ ≈ 0.999
```

Then scale each parameter's update by `1 / √v_t`. Parameters with
consistently large gradients get small steps. Parameters with
consistently small gradients get large steps. The optimiser thus sets
the effective learning rate for each parameter.

---

## 4. Adam = momentum + RMSProp + bias correction

Combining the two:

```
m_t  =  β₁ · m_{t-1}  +  (1 − β₁) · g_t            // 1st moment
v_t  =  β₂ · v_{t-1}  +  (1 − β₂) · g_t²           // 2nd moment
param ←  param  −  lr · m_t / (√v_t + ε)
```

`m` and `v` start at zero, so for the first few steps they are biased
toward zero: `m_1 = (1 − β₁) · g_1` is much smaller than `g_1`. Without
correction, early steps would be very small.

The fix is a **bias correction**:

```
m̂_t  =  m_t  /  (1 − β₁ᵗ)
v̂_t  =  v_t  /  (1 − β₂ᵗ)
param ←  param  −  lr · m̂_t / (√v̂_t + ε)
```

At `t = 1`, the divisors `(1 − β)` cancel the `(1 − β)` factor in the
EMA, so the first-step update is `lr · sign(g)` regardless of gradient
magnitude. As `t` grows, `βᵗ → 0` and the correction factors approach 1,
so the correction disappears and the formula becomes the plain EMA. A
single algebraic factor gives a strong correction early on and none
later.

### What Adam does

The ratio `m / √v` has magnitude roughly `1` whenever the gradient
direction is consistent (signal), regardless of the gradient's absolute
scale. When the gradient is noisy (random direction from step to step),
`m` averages to roughly 0 while `v` stays large, so `m / √v` shrinks.
Adam therefore turns the consistency of the direction into a step size:

- Consistent direction, any magnitude  →  step size ≈ `lr`
- Inconsistent direction  →  step size shrinks toward 0
- The step size does not depend on the raw gradient magnitude

This is why Adam works well across a wide range of problems without
careful learning-rate tuning per parameter.

### Typical hyperparameters

| param | value | what it controls |
|:------|------:|:-----------------|
| `lr`    | 1e-3   | base step size; sometimes 3e-4 for deep nets |
| `β₁`    | 0.9    | EMA decay for the 1st moment (momentum) |
| `β₂`    | 0.999  | EMA decay for the 2nd moment (adaptive lr) |
| `ε`     | 1e-8   | numerical guard added to the denominator (outside the sqrt) |

These defaults often work. The hyperparameter tuned most often is `lr`.

---

## 5. The Optimizer abstraction

Adam needs per-parameter state (`m`, `v`, and `t`), and the same
optimiser instance has to own that state for all of a network's
parameters. The update logic is therefore in an Optimizer, not in the
individual layers.

```cpp
class Optimizer<T> {
    virtual void step(Matrix<T>& param, const Matrix<T>& grad) = 0;
};
```

Stateful optimisers key their state on `&param` (the parameter's
address), with lazy allocation on first use:

```cpp
class Adam : public Optimizer<T> {
    struct State { Matrix<T> m, v; int t = 0; };
    std::unordered_map<Matrix<T>*, State> state_;

    void step(Matrix<T>& W, const Matrix<T>& dW) override {
        State& s = state_[&W];
        if (s.t == 0) { /* lazy init m and v to W's shape */ }
        s.t += 1;
        /* update m, v, then apply bias-corrected step to W */
    }
};
```

A layer's `update` passes its parameters to the optimiser, one at a
time:

```cpp
void Dense::update(Optimizer<T>& opt) override {
    opt.step(W_, dW_);
    opt.step(b_, db_);
}
```

The layer knows what its parameters are; the optimiser decides how they
are updated. Parameter-less layers (activations) inherit a no-op
default. `Network::update` forwards the optimiser to every layer.

### In the training loop

```cpp
SGD<float>  opt(0.1f);                  // or  Adam<float> opt(1e-3f);
for (int epoch = 0; epoch < epochs; ++epoch) {
    for (auto batch : batches) {
        net.forward(X_batch);
        ce.forward(...);
        net.backward(ce.backward());
        net.update(opt);
    }
}
```

Switching optimisers is a one-line change in user code.

---

## 6. The empirical comparison

The optimiser test in `tests/test_optimizer.cpp` minimises a
poorly-conditioned quadratic:

```
f(x, y) = ½ x² + 50 y²        ∇f = (x, 100 y)
```

The gradient in `y` is 100× larger than in `x`. After 100 steps from
`(5, 5)`:

| optimiser           | final loss |
|:--------------------|:-----------|
| SGD (lr = 0.01)     | 1.67       |
| Adam (lr = 0.1)     | 0.08       |

For these learning rates and this step budget, Adam reaches a lower
loss. SGD's single learning rate has to be small
enough not to oscillate on `y`, so `x` hardly moves. Adam adapts the two
axes independently and makes progress on both.

This comparison illustrates the effect of coordinate-wise adaptation;
it does not establish which optimizer is best on another dataset. The
IRIS example keeps SGD, which is sufficient for that small model. The
later examples use Adam as a practical starting point.

![SGD and Adam on the quadratic of this section, 100 steps from (5, 5). Left: the path of each optimiser; the cross marks the minimum. Right: the loss at each step, on a logarithmic scale. SGD brings `y` to 0 in its first step and then moves slowly along `x`. Adam moves both coordinates at the same rate.](figures/10_sgd_adam.png)

---

## 7. Practical recommendations

- **Use Adam at `lr = 1e-3` as an initial experiment.** The examples
  keep its default β1/β2/ε. Check the loss and gradient scale rather
  than assuming these settings will work for every model.
- **Compare optimizers on validation data.** Give each a reasonable
  learning-rate search and a stated training budget. This chapter's
  quadratic does not justify a particular accuracy gain or a rule to
  switch optimizers near the end of training.
- **Treat learning rate as a parameter to investigate.** Adam adapts
  coordinate scales, but it is still sensitive to its global learning
  rate. If training oscillates or diverges, reducing it is one useful
  experiment alongside checking the data, loss, and derivatives.

---

## 8. What MNIST still needs

MNIST still needs:

- **Train/eval mode flag** on Layer, so layers like Dropout can behave
  differently between training and inference.
- **Dropout layer**: zero out a fraction of activations during training,
  pass through during inference.
- **MNIST data loader** for the IDX binary format.

Then we train the MNIST classifier using Adam. That example demonstrates
the full training pipeline; it is not a controlled optimizer comparison.


## Further reading {#sec-10-optimizers-further-reading}

Plain stochastic gradient descent is [@robbins1951sgd]; momentum is
[@polyak1964momentum]; RMSProp was never published as a paper and is
cited by its lecture [@tieleman2012rmsprop]. Adam, which combines the
last two with a bias correction, is [@kingma2015adam].
