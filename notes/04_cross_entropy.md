# 04 — Cross-entropy loss (and the Softmax+CE shortcut)

This note covers the loss function we'll use for every classifier in weft.
It also derives the famous Softmax + cross-entropy "shortcut" — the result
that the combined gradient with respect to the logits is simply `S − T`
(divided by batch size). That identity is the reason almost all classifier
training code looks the way it does.

---

## 1. Why not squared error?

Squared error penalises numeric distance: `(prediction − target)²`. That's
the right loss when the prediction and target are real-valued quantities
(temperature, price, pixel intensity). For classification, our prediction
is a *probability distribution* (Softmax made sure of that), and so is our
target (one-hot, or smoothed). What we want is a loss that measures how
*different* the predicted distribution is from the target distribution.
Squared error doesn't know that the output space is the simplex.

A second reason is practical. Squared error's gradient is bounded by the
size of the prediction-minus-target vector, so a confidently-wrong network
gets only a moderate gradient. We want the opposite: confident wrongness
should be penalised *more* aggressively, not less.

Both problems are solved by cross-entropy, which comes naturally from
information theory and from a maximum-likelihood argument.

---

## 2. Cross-entropy: definition and interpretation

Given a target distribution `t` and a predicted distribution `s` over `K`
classes:

```
H(t, s)  =  − Σᵢ tᵢ · log(sᵢ)
```

The intuition: `−log(sᵢ)` is the **surprise** of seeing class `i` when the
model assigned it probability `sᵢ`. Predict 0.99 on what actually happens
and surprise is tiny (`≈ 0.01`); predict 0.001 and surprise is huge
(`≈ 6.9`). Cross-entropy is the expected surprise, weighted by the *true*
distribution.

For a one-hot target (the typical classifier setup — true class `c` has
`t_c = 1`, others are 0), the sum collapses:

```
H(t, s)  =  − log(s_c)
```

The loss is just the negative log of whatever probability the network
assigned to the correct class. Assign 99% to the truth: ~0.01 loss. Assign
1% to the truth: ~4.6 loss. Assign 0% to the truth (impossible in
practice, but instructive): infinite loss.

> **Equivalent perspective: maximum likelihood.** If your model outputs
> `P(class | input) = sᵢ`, then the likelihood of the data is the product
> of `s_{cₙ}` over examples. Maximising likelihood is equivalent (taking
> logs) to minimising `−Σₙ log(s_{cₙ})`, which is cross-entropy with one-
> hot targets. The information-theoretic and probabilistic stories give
> the same loss.

---

## 3. The batch form

Average over `N` examples:

```
L  =  (1/N) · Σₙ Hₙ(Tₙ, Sₙ)  =  −(1/N) · Σₙ Σᵢ Tₙ[i] · log(Sₙ[i])
```

Averaging (rather than summing) makes the loss magnitude independent of
batch size, so learning rates don't need to be rescaled when you change
the batch.

By the convention we established in `02_backprop.md`, the `1/N` factor
lives **inside the gradient returned by the loss class**, not inside the
backward formulas of layers below it. This keeps layer code simple — they
never see batch-size arithmetic.

---

## 4. The standalone gradient

Differentiating with respect to `Sₙ[i]` (one entry of the predicted
distribution for one example):

```
∂L/∂Sₙ[i]  =  −(1/N) · Tₙ[i] / Sₙ[i]
```

That's what `CrossEntropy::backward()` returns. Note the blow-up when
`Sₙ[i]` is tiny and `Tₙ[i] = 1` — exactly the case of maximum error. The
expression is correct, but it's brittle in float arithmetic.

---

## 5. The Softmax + CE shortcut (the beautiful part)

The Softmax Jacobian (from `03b_jacobian.md`) is
`∂Sᵢ/∂Zⱼ = Sᵢ(δᵢⱼ − Sⱼ)`. Chain it with the CE gradient:

```
∂L/∂Zⱼ  =  Σᵢ  (∂L/∂Sᵢ) · (∂Sᵢ/∂Zⱼ)
        =  Σᵢ  (−Tᵢ / Sᵢ) · Sᵢ · (δᵢⱼ − Sⱼ)         ← Sᵢ cancels!
        =  Σᵢ  (−Tᵢ) · (δᵢⱼ − Sⱼ)
        =  −Tⱼ  +  Sⱼ · Σᵢ Tᵢ
        =  Sⱼ  −  Tⱼ                                ← because Σᵢ Tᵢ = 1
```

Add the `1/N` batch factor and you're done:

```
∂L/∂Z  =  (S − T) / N
```

**The combined gradient through Softmax and cross-entropy is just "subtract
target from prediction, divide by batch size."** The two derivatives, each
of which individually contained pieces that can blow up, telescope to the
simplest possible expression.

There are two payoffs:

- **Pedagogical**: the gradient has an obvious meaning — where your
  predicted probability exceeds the target, push down; where it falls
  short, push up. Magnitude is the discrepancy.
- **Numerical**: no `Sᵢ` ever appears in a denominator in the combined
  form. Underflowing softmax probabilities cause no trouble.

This identity is the reason almost every classification training loop —
in PyTorch, TensorFlow, JAX, here — uses a fused Softmax-cross-entropy
operation internally.

---

## 6. Why we still implement them separately in weft

For pedagogy, we keep `Softmax` and `CrossEntropy` as separate classes,
each with its own `backward`. The standalone formulas are correct and
match the math we derived in `03_activations.md`. The composition
`Softmax.backward(CE.backward())` is exactly `(S − T)/N` — that is the
identity our test 4 verifies numerically. For IRIS-scale problems the
numerical fragility of the standalone form never manifests. When we
eventually need the fused form for stability (large-K classifiers,
extreme logit values), we'll add a `SoftmaxCrossEntropy` convenience
class. The mathematics is the same either way.

---

## 7. What we built

`src/Loss.h` — abstract base. Two methods: `forward(predicted, target) ->
scalar` (caches its inputs) and `backward() -> dL/d(predicted)`. The
mean-over-batch `1/N` factor lives inside the gradient by convention.

`src/CrossEntropy.h` — concrete loss:

- `forward`: computes the average cross-entropy. A tiny `eps` is added
  inside `log` for safety against an exact zero in `predicted`.
- `backward`: computes `−(1/N) · T / S` elementwise (with the same eps in
  the denominator).

`tests/test_crossentropy.cpp` — five tests:

1. Forward on a hand-checkable single example: `L = −log(0.7)`.
2. Forward batch averaging.
3. Numerical gradient check on `dL/dS`.
4. **The combined identity**: `Softmax.backward(CE.backward()) == (S − T)/N`,
   computed two ways and compared. This is mathematical proof in code.
5. Full-chain numerical gradient check: perturb each logit `Z[i,j]`,
   confirm `(L(Z+ε) − L(Z−ε))/(2ε)` matches the `dZ` produced by chaining
   `Softmax.backward(CE.backward())`.

All 35 tests in the project (Matrix, Dense, ReLU, Softmax, CrossEntropy)
pass.

---

## 8. What's next

We have every piece of a classifier:

```
Dense  →  ReLU  →  Dense  →  Softmax  →  CrossEntropy
```

The next session wires these into a small `Network` class that holds an
ordered list of layers and runs them forward/backward as a unit, then
trains it on the **IRIS** dataset — the first time we'll watch loss curves
drop on real data.
