# 07 — Optimizers (SGD and Adam)

So far every layer's update method has been plain stochastic gradient
descent baked into the layer itself. Adam (and friends) need
per-parameter state, so we need a cleaner separation. This note covers
the refactor, the two ideas behind Adam, and why it usually wins.

---

## 1. Why plain SGD struggles

Plain SGD updates `param ← param − lr · g`. Two structural weaknesses:

**One learning rate for every parameter.** Gradient magnitudes can vary
by orders of magnitude across parameters of the same network — early
layers vs. late layers, weights vs. biases, parameters that interact with
sparse vs. dense features. A single learning rate has to be small enough
not to blow up on the largest gradients, which means everything else
moves too slowly.

**No memory between steps.** Each step uses only the current minibatch's
gradient. A consistent signal pointing one way step after step gets the
same treatment as a noisy oscillation; SGD treats every gradient as
authoritative even if it's mostly variance.

Adam fixes each with a different idea, then combines them.

---

## 2. Momentum — accelerate in consistent directions

Track an exponential moving average of the gradient:

```
m_t  =  β₁ · m_{t-1}  +  (1 − β₁) · g_t      β₁ ≈ 0.9
```

`m` is the recent average gradient direction. Steps in consistent
directions accumulate; oscillating directions cancel and `m` stays small.

Then update with `m_t` instead of `g_t`. Conceptually it gives the
optimiser inertia — in a long valley you accelerate, on a noisy plateau
you don't get yanked around by individual minibatches.

---

## 3. RMSProp — per-parameter step sizes

Track an exponential moving average of *squared* gradients, per
parameter:

```
v_t  =  β₂ · v_{t-1}  +  (1 − β₂) · g_t²     β₂ ≈ 0.999
```

Then scale each parameter's update by `1 / √v_t`. Parameters with
consistently *large* gradients get *small* steps. Parameters with
consistently *small* gradients get *large* steps. The optimiser
self-calibrates the effective learning rate per parameter.

---

## 4. Adam = momentum + RMSProp + bias correction

Combine them:

```
m_t  =  β₁ · m_{t-1}  +  (1 − β₁) · g_t            // 1st moment
v_t  =  β₂ · v_{t-1}  +  (1 − β₂) · g_t²           // 2nd moment
param ←  param  −  lr · m_t / (√v_t + ε)
```

There's one subtlety. `m` and `v` start at zero, so for the first few
steps they're biased toward zero — `m_1 = (1 − β₁) · g_1` is much smaller
than `g_1`. Without correction, early steps would be tiny.

The fix is a **bias correction**:

```
m̂_t  =  m_t  /  (1 − β₁ᵗ)
v̂_t  =  v_t  /  (1 − β₂ᵗ)
param ←  param  −  lr · m̂_t / (√v̂_t + ε)
```

At `t = 1`, the divisors `(1 − β)` cancel the `(1 − β)` factor in the
EMA, so the first-step update is `lr · sign(g)` regardless of gradient
magnitude. As `t` grows, `βᵗ → 0` and the correction factors approach 1
— corrections fade out and the formula becomes the plain EMA. Beautifully
clean: a single algebraic factor that smoothly transitions from
"bias-correct hard early on" to "trust the EMA later."

### What Adam intuitively does

A concrete way to see why Adam is well-behaved: `m / √v` has magnitude
roughly `1` whenever the gradient direction is consistent (signal),
regardless of the gradient's absolute scale. When the gradient is noisy
(random direction step to step), `m` averages to roughly 0 while `v`
stays large, so `m / √v` shrinks. So **Adam effectively translates "how
consistent is this direction?" into a step size**:

- Consistent direction, any magnitude  →  step size ≈ `lr`
- Inconsistent direction  →  step size shrinks toward 0
- Decoupled from raw gradient magnitude

This is why Adam works well across a huge range of problems without
careful learning-rate tuning per parameter.

### Typical hyperparameters

| param | value | what it controls |
|:------|------:|:-----------------|
| `lr`    | 1e-3   | base step size; sometimes 3e-4 for deep nets |
| `β₁`    | 0.9    | EMA decay for the 1st moment (momentum) |
| `β₂`    | 0.999  | EMA decay for the 2nd moment (adaptive lr) |
| `ε`     | 1e-8   | numerical guard inside the sqrt |

These defaults work surprisingly often. The most-tuned knob is `lr`.

---

## 5. The architectural refactor

The old `Layer::update(T learning_rate)` baked the SGD step into each
layer. Adam needs per-parameter state (`m`, `v`, `t`), and the same
optimiser instance has to own the state for *all* of a network's
parameters — so the update logic can't live inside individual layers
anymore.

New shape:

```cpp
class Optimizer<T> {
    virtual void step(Matrix<T>& param, const Matrix<T>& grad) = 0;
};
```

Stateful optimisers key their state on `&param` (the parameter's
address), with lazy allocation on first contact:

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

Each layer's `update` becomes "here are my parameters, please update
them":

```cpp
void Dense::update(Optimizer<T>& opt) override {
    opt.step(W_, dW_);
    opt.step(b_, db_);
}
```

The layer encapsulates *what* its parameters are; the optimiser decides
*how* they get updated.  Parameter-less layers (activations) inherit a
no-op default.

`Network::update` just forwards the optimiser to every layer.

### Training-loop syntax now

```cpp
SGD<float>  opt(0.1f);                  // or Adam<float> opt(1e-3f);
for (int epoch = 0; epoch < epochs; ++epoch) {
    for (auto batch : batches) {
        net.forward(X_batch);
        ce.forward(...);
        net.backward(ce.backward());
        net.update(opt);                // <-- this is the only change
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

22× lower loss. SGD's single learning rate has to be conservative enough
not to oscillate on `y`, which leaves `x` essentially frozen. Adam
adapts both axes independently and handles them at once.

On IRIS the difference is less dramatic — the problem is well-scaled and
small enough that SGD does fine — but Adam reaches lower training loss
faster (epoch 100 train loss: SGD 0.05, Adam 0.02). Both hit ~96.7% test
accuracy. The real payoff for Adam shows up on larger, less well-behaved
problems; MNIST will start to show the difference more, and a ConvNet on
ImageNet-style data would show it dramatically.

---

## 7. Practical recommendations

- **Start with Adam at `lr = 1e-3`.** Default β1/β2/ε. Works almost
  always for an initial trained model.
- **Switch to SGD with a tuned schedule for the last few epochs** if
  you're squeezing out the final 0.5% of accuracy. Plain SGD + decay
  often generalises a touch better at the end of training (a real
  empirical finding, see e.g. "SWA" — Stochastic Weight Averaging).
  We're not at this level of polish yet.
- **The `lr` for Adam is much less sensitive than for SGD** because the
  adaptive part normalises gradient magnitudes. The reciprocal-of-this
  insight is also true: if Adam is going wrong, the learning rate is
  almost always the first thing to halve.

---

## 8. What's next

The pieces we're missing for MNIST:

- **Train/eval mode flag** on Layer, so layers like Dropout can behave
  differently between training and inference.
- **Dropout layer**: zero out a fraction of activations during training,
  pass through during inference.
- **MNIST data loader** for the IDX binary format.

Then we train the MNIST classifier, which is where Adam over SGD will
start mattering visibly.
