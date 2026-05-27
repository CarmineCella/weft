# 06 — IRIS: the first end-to-end classifier

This note walks through the IRIS example end-to-end and the new tools the
library acquired to make it possible.

The example lives at `examples/iris_classifier/`. The library tools it
relies on are:

- `Matrix::selectColumns(indices)` — slice a matrix by column indices.
- `Standardizer` — fit per-feature mean/std on training data, apply to
  both train and test.
- `one_hot`, `train_test_split`, `shuffled_indices`, `accuracy` — free
  functions in `Data.h`.

---

## 1. The standard ML pipeline

Almost every supervised-learning example follows the same six steps. IRIS
is the smallest interesting instance of this pattern; MNIST, the audio
classifier, and the autoencoder will follow the same skeleton.

1. **Load** the raw data into matrices (features × examples).
2. **One-hot encode** the class labels into a `(num_classes × N)` target
   matrix.
3. **Split** into train/test (random, with a fixed seed).
4. **Standardise** features — fit the scaler on TRAIN ONLY, then apply to
   both splits.
5. **Train** with mini-batch SGD, shuffling each epoch.
6. **Evaluate** with accuracy on the held-out test set.

The last four steps are essentially mechanical once the data is loaded.
The library's `Data.h` and `Standardizer.h` cover them.

---

## 2. Why standardise features?

Raw IRIS features have very different scales: sepal length is in `[4.3, 7.9]`,
petal width is in `[0.1, 2.5]`. A gradient update step has the same
learning rate for every weight, so a weight attached to a large-scale
feature receives a much larger raw gradient than one attached to a
small-scale feature, simply because of the input magnitudes. This forces
the optimiser to take small steps to avoid blowing up the large-scale
features, and learning crawls on everything else.

Standardising — subtract per-feature mean, divide by per-feature std —
puts all features on roughly the same scale (`mean ≈ 0`, `std ≈ 1`).
Suddenly every weight sees comparable inputs and a single learning rate
works well across all of them.

This generalises:

- **Images**: typically scale pixels to `[0, 1]` or standardise per
  channel.
- **Audio**: often log-mel spectrograms then standardise.
- **Tabular**: standardise numeric features; one-hot encode categoricals.

`Standardizer` is the simplest case; we'll add specialised preprocessors
later if needed.

---

## 3. The "fit on train only" rule

The single most common subtle bug in ML pipelines is computing the mean
and std on the *entire* dataset (train + test together) and using those
statistics to standardise both splits. This is called **data leakage**:
statistics of the test set bleed into preprocessing, which lets the
training process implicitly "see" test data. The test accuracy ends up
optimistic and you don't notice until production.

`Standardizer`'s API encourages the right pattern:

```cpp
Standardizer<float> sc;
Matrix<float> X_train_std = sc.fit_transform(X_train);   // learn AND apply
Matrix<float> X_test_std  = sc.transform(X_test);        // apply (same params)
```

`fit_transform` learns mean/std from the data it's given; `transform`
just applies the learned parameters. We never call `fit` on the test
data. The same scaler instance is used for both, so the test inputs are
mapped into the same space the training inputs live in.

---

## 4. The training loop pattern

Mini-batch SGD with per-epoch shuffling:

```cpp
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
        net.update(learning_rate);
    }
}
```

Three things matter here:

- **Mini-batches** (rather than the full dataset or single examples).
  Full-batch gradient descent is too memory-hungry for big datasets and
  has fewer updates per epoch; pure stochastic gradient descent (one
  example at a time) has too noisy a gradient. Mini-batches are the
  middle ground — 16, 32, 64 are typical for small problems; 128–512 for
  larger ones.
- **Shuffle every epoch**. Without it, the network sees the same batches
  in the same order every time. Even though SGD is "stochastic" already
  (because each batch is just a noisy estimate of the true gradient),
  the noise from a deterministic batch order is correlated across
  epochs, which makes convergence slower and can settle the optimiser
  into bad local patterns.
- **Use the same `shuffled_indices` seed strategy across runs** so
  results are reproducible. The example uses the epoch number as the
  seed, which gives different orders per epoch but is fully
  deterministic.

---

## 5. Accuracy as the eval metric

For classification, the natural evaluation metric is the fraction of
examples where the model's predicted class equals the true class. That's
just `argmax-per-column` of predictions vs `argmax-per-column` of
one-hot targets:

```cpp
T accuracy(predictions, targets):
    correct = 0
    for each column j:
        if argmax(predictions[:, j]) == argmax(targets[:, j]):
            correct += 1
    return correct / N
```

The implementation in `Data.h` works whether `predictions` is post-softmax
probabilities or raw logits — only the ranking of values matters.

Loss and accuracy are not the same thing! Loss is a continuous, smooth
measure used for gradient descent. Accuracy is what you actually care
about. They usually move together, but it's possible for loss to keep
going down (the model becomes more confidently right where it was
already right) while accuracy plateaus.

---

## 6. What the example actually shows

```
loaded 150 examples, 4 features, 3 classes
split: 120 train, 30 test  (seed=1)
features standardised (mean ~ 0, std ~ 1 on train)

architecture: Dense(4,16) -> ReLU -> Dense(16,3) -> Softmax
loss:         cross-entropy
optimiser:    plain SGD, lr=0.1, batch_size=16
training for 100 epochs

  epoch   train_loss   train_acc   test_loss   test_acc
  -----   ----------   ---------   ---------   --------
      0       1.2618      0.4250      1.2937     0.4000
     10       0.2062      0.9500      0.1208     1.0000
     20       0.1320      0.9667      0.0751     1.0000
     ...
    100       0.0506      0.9917      0.0611     0.9667

final test accuracy: 0.9667  (29/30 correct)
```

Two things to look at:

- The model's untrained baseline is ~40% accuracy. Random guessing would
  give ~33% (1/3 classes). The slight advantage at epoch 0 is just luck
  in how the random initial weights happened to split the inputs.
- The gap between final train accuracy (99.2%) and test accuracy (96.7%)
  is a mild form of overfitting. With 120 training examples and a network
  of ~80 parameters, the network has enough capacity to memorise some
  examples that don't generalise. For IRIS this is harmless; for MNIST it
  will be the central practical problem we have to solve, with techniques
  like Dropout, weight decay, and early stopping.

---

## 7. The seed-variance warning

The seed used to split train/test changes the result more than you might
expect:

| seed | test accuracy |
|-----:|:--------------|
|    0 | 0.9333 |
|    1 | 0.9667 |
|    7 | 0.9333 |
|   42 | 0.8667 |
|  123 | 1.0000 |

Same model, same training, same data — just a different 30 examples in
the test set. On small test sets a single accuracy number is essentially
noise. Two implications:

- Never report a single test accuracy from a tiny test set without
  noting the variance.
- For real evaluation, use **k-fold cross-validation**: split the data
  into k disjoint folds, train k models leaving one fold out each time,
  and average the per-fold accuracies. Reduces variance and uses every
  example for both training and evaluation. We'll add a `kfold` helper
  in `Data.h` when we hit problems where this variance actually obscures
  real differences.

---

## 8. What's next

Now that we have a working classifier, the next milestone scales up:
**MNIST** (handwritten digits). 60,000 training images × 784 pixels × 10
classes. Three things that IRIS got away with will start to bite:

- **Plain SGD becomes painful**: gradients across thousands of parameters
  need adaptive learning rates. → **Adam** (and an `Optimizer` refactor).
- **Capacity outstrips the data more aggressively**, so we'll need
  regularisation. → **Dropout** (and a training/eval mode flag on
  `Layer`).
- **Single accuracy numbers stop being noisy** (10,000 test examples is
  plenty), so we can skip k-fold for now.

After MNIST: orchestral instrument sounds, then the autoencoder, then
ConvNets.
