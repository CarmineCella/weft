# 7 — Cross-entropy loss (and the Softmax+CE shortcut)

This chapter covers the loss function we use for every classifier in
weft. It also derives the Softmax + cross-entropy shortcut: the combined
gradient with respect to the logits is `S − T`, divided by the batch
size. Almost all classifier training code is built on that identity.

---

## 1. Why not squared error?

Squared error penalises numeric distance: `(prediction − target)²`. That
is the right loss when the prediction and target are real-valued
quantities (temperature, price, pixel intensity). For classification, the
prediction is a probability distribution (Softmax guarantees that), and
so is the target (one-hot, or smoothed). We need a loss that measures how
different the predicted distribution is from the target distribution.
Squared error does not take into account that the output space is the
simplex.

The second reason is practical. Squared error's gradient is bounded by
the size of the prediction-minus-target vector, so a confidently wrong
network gets only a moderate gradient. We need the opposite: a confident
wrong prediction should be penalised more, not less.

Cross-entropy solves both problems. It follows from information theory
and from a maximum-likelihood argument.

---

## 2. Cross-entropy: definition and interpretation

Given a target distribution `t` and a predicted distribution `s` over `K`
classes:

```
H(t, s)  =  − Σᵢ tᵢ · log(sᵢ)
```

The quantity `−log(sᵢ)` is the **surprise** of seeing class `i` when the
model assigned it probability `sᵢ`. If the model predicts 0.99 for the
class that occurs, the surprise is small (`≈ 0.01`). If it predicts
0.001, the surprise is large (`≈ 6.9`). Cross-entropy is the expected
surprise, weighted by the true distribution.

For a one-hot target (the usual classifier setup, where the true class
`c` has `t_c = 1` and the others are 0), the sum reduces to one term:

```
H(t, s)  =  − log(s_c)
```

The loss is the negative log of the probability the network assigned to
the correct class. With 99% on the correct class the loss is ~0.01. With
1% it is ~4.6. With 0% (impossible in practice, but instructive) it is
infinite.

> **Equivalent perspective: maximum likelihood.** If your model outputs
> `P(class | input) = sᵢ`, then the likelihood of the data is the product
> of `s_{cₙ}` over examples. Maximising likelihood is equivalent (taking
> logs) to minimising `−Σₙ log(s_{cₙ})`, which is cross-entropy with
> one-hot targets. The information-theoretic and the probabilistic
> arguments give the same loss.

![The loss on one example as a function of the probability the network gives to the correct class. Cross-entropy grows without limit as that probability approaches 0. The squared error never exceeds 1.](figures/07_cross_entropy.png)

---

## 3. The batch form

Average over `N` examples:

```
L  =  (1/N) · Σₙ Hₙ(Tₙ, Sₙ)  =  −(1/N) · Σₙ Σᵢ Tₙ[i] · log(Sₙ[i])
```

Averaging (rather than summing) makes the loss magnitude independent of
batch size, so learning rates don't need to be rescaled when the batch
size changes.

By the convention of [chapter 3](03_backprop.md), the `1/N` factor is
part of the gradient returned by the loss class, not of the backward
formulas of the layers below it. This keeps layer code simple: layers
never handle batch-size arithmetic.

---

## 4. The standalone gradient

Differentiating with respect to `Sₙ[i]` (one entry of the predicted
distribution for one example):

```
∂L/∂Sₙ[i]  =  −(1/N) · Tₙ[i] / Sₙ[i]
```

This is what `CrossEntropy::backward()` returns. It grows without bound
when `Sₙ[i]` is very small and `Tₙ[i] = 1`, which is the case of maximum
error. The expression is correct, but it is fragile in float arithmetic.

---

## 5. The Softmax + CE shortcut

The Softmax Jacobian (from [chapter 6](06_jacobian.md)) is
`∂Sᵢ/∂Zⱼ = Sᵢ(δᵢⱼ − Sⱼ)`. Chain it with the CE gradient:

```
∂L/∂Zⱼ  =  Σᵢ  (∂L/∂Sᵢ) · (∂Sᵢ/∂Zⱼ)
        =  Σᵢ  (−Tᵢ / Sᵢ) · Sᵢ · (δᵢⱼ − Sⱼ)         ← Sᵢ cancels!
        =  Σᵢ  (−Tᵢ) · (δᵢⱼ − Sⱼ)
        =  −Tⱼ  +  Sⱼ · Σᵢ Tᵢ
        =  Sⱼ  −  Tⱼ                                ← because Σᵢ Tᵢ = 1
```

Adding the `1/N` batch factor gives:

```
∂L/∂Z  =  (S − T) / N
```

The combined gradient through Softmax and cross-entropy is the
prediction minus the target, divided by the batch size. Each of the two
derivatives contains terms that can grow without bound; in the product
these terms cancel.

The combined form has these advantages:

- **Pedagogical**: the gradient has a direct meaning. Where the
  predicted probability exceeds the target, the update pushes it down;
  where it falls short, the update pushes it up. The magnitude is the
  discrepancy.
- **Numerical**: no `Sᵢ` ever appears in a denominator in the combined
  form. Softmax probabilities that underflow cause no problem.

Because of this identity, almost every classification training loop (in
PyTorch, TensorFlow, JAX) uses a fused Softmax-cross-entropy operation
internally. weft keeps the two separate, for the reasons in section 6.

---

## 6. Why we still implement them separately in weft

For pedagogy, we keep `Softmax` and `CrossEntropy` as separate classes,
each with its own `backward`. The standalone formulas are correct and
match the math we derived in [chapter 5](05_activations.md). The
composition `Softmax.backward(CE.backward())` equals `(S − T)/N`, and
test 4 verifies this identity numerically. For IRIS-scale problems the
numerical fragility of the standalone form never appears. A library
meant for large-K classifiers or extreme logit values would add a fused
`SoftmaxCrossEntropy` class for stability; this book never needs one.
The mathematics is the same either way.

---

## 7. What we built

`src/Loss.h` is the abstract base, with `forward(predicted, target) ->
scalar` (caches its inputs) and `backward() -> dL/d(predicted)`. By
convention, the mean-over-batch `1/N` factor is part of the gradient.

`src/CrossEntropy.h` is the concrete loss:

- `forward`: computes the average cross-entropy. A small `eps` is added
  inside `log` to guard against an exact zero in `predicted`.
- `backward`: computes `−(1/N) · T / S` elementwise (with the same eps in
  the denominator).

`tests/test_crossentropy.cpp` contains these tests:

1. Forward on a hand-checkable single example: `L = −log(0.7)`.
2. Forward batch averaging.
3. Numerical gradient check on `dL/dS`.
4. **The combined identity**: `Softmax.backward(CE.backward()) == (S − T)/N`,
   computed two ways and compared.
5. Full-chain numerical gradient check: perturb each logit `Z[i,j]`,
   confirm `(L(Z+ε) − L(Z−ε))/(2ε)` matches the `dZ` produced by chaining
   `Softmax.backward(CE.backward())`.

All 35 checks in the five test programs so far (Matrix, Dense, ReLU, Softmax, CrossEntropy)
pass.

---

## 8. The classifier so far

We now have every component of a classifier:

```
Dense  →  ReLU  →  Dense  →  Softmax  →  CrossEntropy
```

The next chapter combines these in a small `Network` class that holds an
ordered list of layers and runs them forward and backward as a unit.
Then [chapter 9](09_iris.md) trains it on the IRIS dataset, the first
training run on real data.


## Further reading

Entropy and cross-entropy as measures of information come from
[@shannon1948communication]. The pairing of a softmax output with a
cross-entropy loss, and the probabilistic argument for it, is in
[@bridle1990softmax].
