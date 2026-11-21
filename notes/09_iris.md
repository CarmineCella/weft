# 9 — IRIS: the first end-to-end classifier

This chapter goes through the IRIS example from start to finish, and the
tools added to the library for it.

The example is in `examples/iris_classifier.cpp`. It uses these library
tools:

- `Matrix::selectColumns(indices)`: slices a matrix by column indices.
- `Standardizer`: fits per-feature mean/std on training data and applies
  them to both train and test.
- `one_hot`, `train_test_split`, `shuffled_indices`, `accuracy`: free
  functions in `Data.h`.

---

## 1. The standard ML pipeline

Almost every supervised-learning example follows the same steps. IRIS
is the smallest instance of this pattern. MNIST, the audio classifier,
and the autoencoder follow the same structure.

1. **Load** the raw data into matrices (features × examples).
2. **One-hot encode** the class labels into a `(num_classes × N)` target
   matrix.
3. **Split** into train/test (random, with a fixed seed).
4. **Standardise** features: fit the scaler on the training split only,
   then apply it to both splits.
5. **Train** with mini-batch SGD, shuffling each epoch.
6. **Evaluate** with accuracy on the held-out test set.

Once the data is loaded, the last four steps are mechanical. The
library's `Data.h` and `Standardizer.h` cover them.

---

## 2. Why standardise features?

Raw IRIS features have very different scales: sepal length is in `[4.3, 7.9]`,
petal width is in `[0.1, 2.5]`. A gradient update step has the same
learning rate for every weight, so a weight attached to a large-scale
feature receives a much larger raw gradient than one attached to a
small-scale feature, only because of the input magnitudes. The optimiser
must then take small steps so that the weights on the large-scale
features do not diverge, and learning is slow for all the others.

Standardising (subtracting the per-feature mean and dividing by the
per-feature std) puts all features on roughly the same scale
(`mean ≈ 0`, `std ≈ 1`). Every weight then receives comparable inputs,
and a single learning rate works well for all of them.

The same applies to other data:

- **Images**: typically scale pixels to `[0, 1]` or standardise per
  channel.
- **Audio**: often log-mel spectrograms then standardise.
- **Tabular**: standardise numeric features; one-hot encode categoricals.

`Standardizer` is the simplest case. We add specialised preprocessors
later if needed.

---

## 3. The "fit on train only" rule

The most common subtle bug in ML pipelines is computing the mean and std
on the entire dataset (train and test together) and using those
statistics to standardise both splits. This is called **data leakage**:
statistics of the test set enter the preprocessing, so the training
process implicitly uses test data. The test accuracy is then optimistic,
and the error goes unnoticed until production.

`Standardizer`'s API encourages the right pattern:

```cpp
Standardizer<float> sc;
Matrix<float> X_train_std = sc.fit_transform(X_train);   // learn AND apply
Matrix<float> X_test_std  = sc.transform(X_test);        // apply (same params)
```

`fit_transform` learns mean/std from the data it's given; `transform`
only applies the learned parameters. We never call `fit` on the test
data. The same scaler instance is used for both, so the test inputs are
mapped into the same space as the training inputs.

---

## 4. The training loop pattern

Mini-batch SGD with per-epoch shuffling:

```cpp
SGD<float> opt(learning_rate);
for (int epoch = 0; epoch < epochs; ++epoch) {
    auto idx = shuffled_indices(X_train.cols(), epoch);   // new order each epoch

    for (size_t start = 0; start < X_train.cols(); start += batch_size) {
        size_t end = min(start + batch_size, X_train.cols());
        vector<size_t> batch_idx(idx.begin() + start, idx.begin() + end);

        Matrix<float> X_batch = X_train.selectColumns(batch_idx);
        Matrix<float> Y_batch = Y_train.selectColumns(batch_idx);

        Matrix<float> S = net.forward(X_batch);
        ce.forward(S, Y_batch);
        net.backward(ce.backward());
        net.update(opt);
    }
}
```

The important choices are:

- **Mini-batches** (rather than the full dataset or single examples).
  Full-batch gradient descent needs too much memory for big datasets
  and makes fewer updates per epoch. Pure stochastic gradient descent
  (one example at a time) has too noisy a gradient. Mini-batches are in
  between: 16, 32, 64 are typical for small problems, 128–512 for
  larger ones.
- **Shuffle every epoch**. Without it, the network sees the same batches
  in the same order every time. SGD is already stochastic, because each
  batch gives a noisy estimate of the true gradient, but the noise from
  a deterministic batch order is correlated across epochs. This slows
  convergence and can settle the optimiser into bad local patterns.
- **Use the same `shuffled_indices` seed strategy across runs** so
  the batch order is reproducible. The example uses the epoch number as
  the seed, which gives different orders per epoch but the same orders
  on every run. (The weight initialisation is not seeded, so final
  numbers still vary from run to run; see section 7.)

---

## 5. Accuracy as the eval metric

For classification, the natural evaluation metric is the fraction of
examples where the model's predicted class equals the true class. It
compares the `argmax-per-column` of the predictions with the
`argmax-per-column` of the one-hot targets:

```cpp
T accuracy(predictions, targets):
    correct = 0
    for each column j:
        if argmax(predictions[:, j]) == argmax(targets[:, j]):
            correct += 1
    return correct / N
```

The implementation in `Data.h` works whether `predictions` is post-softmax
probabilities or raw logits, because only the ranking of the values
matters.

Loss and accuracy are not the same thing. Loss is a continuous, smooth
measure used for gradient descent. Accuracy is the quantity of interest.
They usually move together, but the loss can keep decreasing (the model
becomes more confidently right where it was already right) while
accuracy stays flat.

---

## 6. What the example shows

```
loaded 150 examples, 4 features, 3 classes
split: 120 train, 30 test  (seed=1)
features standardised (mean ~ 0, std ~ 1 on train)

architecture:
    Dense(4, 16)
    ReLU
    Dense(16, 3)
    Softmax
loss:         cross-entropy
optimiser:    plain SGD, lr=0.1, batch_size=16
training for 100 epochs

  epoch   train_loss   train_acc   test_loss   test_acc
  -----   ----------   ---------   ---------   --------
      0       1.7867      0.1917      1.8254     0.2333
     10       0.2084      0.9250      0.1847     1.0000
     20       0.1401      0.9500      0.1188     0.9667
     ...
    100       0.0588      0.9833      0.0476     1.0000

final test accuracy: 1.0000  (30/30 correct)
```

In this output:

- The untrained network is at 23% accuracy, below the 33% that random
  guessing among three classes would give. This has no significance: it
  is the result of the random initial weights. Ten epochs later it is
  at 100%.
- Your numbers will differ. The split and the batch order are seeded,
  but the initial weights are not, so every run starts from different
  weights. This is deliberate, and the next section discusses the
  consequences.

---

## 7. The seed-variance warning

Over twenty runs of the example, the final test accuracy takes several
values:

| test accuracy | runs out of 20 |
|:--------------|---------------:|
| 30/30 = 1.0000 | 15 |
| 29/30 = 0.9667 |  4 |
| 28/30 = 0.9333 |  1 |

The model, the data, the split and the batch order are the same; only
the random initial weights differ. With 30 test examples, one example is
3.3 percentage points, and a difference of one or two examples between
runs is ordinary. A different train/test split moves the number at
least as much. On a small test set a single accuracy number is mostly
noise. Therefore:

- Never report a single test accuracy from a tiny test set without
  noting the variance.
- For real evaluation, use **k-fold cross-validation**: split the data
  into k disjoint folds, train k models leaving one fold out each time,
  and average the per-fold accuracies. This reduces variance and uses
  every example for both training and evaluation. The library does not
  have a `kfold` helper; the larger datasets of the following chapters
  have test sets big enough not to need one.

---

## 8. From IRIS to MNIST

The next example is larger: MNIST (handwritten digits), with 60,000
training images × 784 pixels × 10 classes. It needs additions that IRIS
did not:

- **Regularisation.** The model capacity exceeds the data by a larger
  margin, so we need Dropout (and a training/eval mode flag on
  `Layer`).
- **A faster optimiser.** Plain SGD is slower than Adam on a problem
  this size. The next chapter adds Adam; its practical advantage
  appears on MNIST.

We can skip k-fold cross-validation: with 10,000 test examples, a single
accuracy number is no longer noisy.

After MNIST come orchestral instrument sounds, then the autoencoder,
then ConvNets.


## Further reading

The Iris measurements were published, with the first statistical
classifier applied to them, by [@fisher1936iris].
