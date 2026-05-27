# 09 — MNIST classifier

MNIST is the canonical "is your neural network actually working" test —
60,000 training and 10,000 test images of 28×28-pixel handwritten digits,
labelled 0-9. It's small enough to train on a laptop CPU in minutes, and
big enough that the regularisation and optimisation choices we made in
the last few sessions actually matter.

This is the first weft example that exercises every piece of the library
in one go: the IDX loader, an MLP with hidden Dense layers, ReLU,
Dropout, Softmax, cross-entropy, Adam, and train/eval mode switching.

---

## 1. The data pipeline

```cpp
X_train = mnist::load_images<float>(data_dir + "/train-images-idx3-ubyte");
y_train = mnist::load_labels       (data_dir + "/train-labels-idx1-ubyte");
Y_train = one_hot<float>(y_train, 10);
```

`load_images` returns a `(784, 60000)` matrix — one column per image,
pixel values normalised to `[0, 1]`. This is already a reasonable scale
for He-initialised Dense layers, so we skip the Standardizer. (The
canonical PyTorch tutorial normalises to mean 0.1307, std 0.3081, but
the difference in final accuracy is tiny and the simpler pipeline is
worth keeping.)

`load_labels` returns a `vector<int>` of digit indices. `one_hot<float>`
turns that into a `(10, 60000)` matrix with a single 1 per column.

---

## 2. The architecture

```
Dense(784, 256) -> ReLU -> Dropout(0.3)
Dense(256, 128) -> ReLU -> Dropout(0.3)
Dense(128, 10)  -> Softmax
```

About 235,000 parameters — large enough relative to 60,000 training
examples that overfitting is a real risk, which is exactly the regime
where dropout starts paying off.

**Why two hidden layers, not one or five?** A single hidden layer can in
principle approximate any continuous function (universal approximation
theorem), but in practice deep narrow networks compose features more
efficiently than wide shallow ones. Two hidden layers is enough to get
MNIST to ~97-98% without going deep enough that vanishing-gradient
issues or training instability start to matter.

**Why 256 -> 128, not 256 -> 256 or 128 -> 64?** Powers of two are a
historical convention (originally cache- and SIMD-friendly), and a
shrinking pyramid is a sensible default — early layers extract many
low-level features, later layers compress them into class-relevant
ones. The exact widths are not load-bearing; ±2x in any of them changes
final accuracy by a fraction of a percent.

**Why Dropout 0.3 instead of the classic 0.5?** 0.5 is what Srivastava's
2014 paper used on much larger fully-connected networks. With our
modest width, 0.3 regularises enough without slowing convergence too
much. 0.2 or 0.4 would work nearly as well.

**No dropout on the input or the classification head.** Dropout right
before the softmax would directly zero out class logits, which is
nonsensical. Dropout on raw pixels is an option but a weak one —
"random crop"-style augmentation is a better fit for the input.

---

## 3. The optimiser choice

We use Adam, not SGD. We could use SGD with a careful learning rate
schedule and probably hit similar accuracy, but Adam is dramatically
less fussy to tune. With a single hyperparameter (lr = 1e-3) and no
schedule, it converges to ~97% test accuracy in ten epochs.

Plain SGD at the same learning rate would either diverge or crawl,
depending on the value. SGD with momentum at lr ≈ 0.01 with a learning
rate decay would be competitive, but you'd have two hyperparameters
to set well instead of one. The point of Adam for this kind of
problem is exactly this: less time tuning, more time understanding.

---

## 4. The training loop in detail

```cpp
for (int epoch = 1; epoch <= epochs; ++epoch) {
    net.train();                                  // (a)
    auto idx = shuffled_indices(X.cols(), epoch);

    for (mini-batches) {
        Matrix S = net.forward(X_batch);
        float  L = ce.forward(S, Y_batch);
        net.backward(ce.backward());
        net.update(opt);
    }

    net.eval();                                   // (b)
    auto [test_loss, test_acc] = evaluate(...);
}
```

The two mode switches are the new piece compared to the IRIS example.

`(a)` puts the network into training mode. Dropout fires, randomly
zeroing 30% of each hidden activation. This is the entire point — we
deliberately corrupt the activations to force the network to be
redundant, knowing the test pass will get the cleaner signal.

`(b)` puts the network into inference mode. Dropout becomes the
identity. The network we evaluate is the full, un-corrupted one.

Without `(b)`, the test accuracy reported each epoch would be measured
on a randomly-corrupted network — much lower, and not representative of
the model the user actually wants to deploy.

**Why mini-batches at this size?** Batch 128 is a sweet spot for MNIST.
Larger batches (512+) get less stochastic noise per update, which can
help final accuracy slightly but means fewer updates per epoch — and
modern wisdom is that the noise actually helps generalisation, so the
trade-off is roughly even. Smaller batches (16) waste time on
per-iteration overhead and don't vectorise well.

---

## 5. Train vs test accuracy: reading the gap

The reported "train acc" in our table is the running accuracy
*during the training pass* — that is, measured with dropout still
active. The "test acc" is measured in eval mode.

It's common, especially early in training, for **test accuracy to
exceed train accuracy**. This isn't a bug — it's exactly what dropout
is designed to do. The network has to make correct predictions despite
having 30% of its activations zeroed; the test pass gives it the full
network and it does noticeably better.

Later in training, as the network starts to overfit, train accuracy
(even with dropout) will catch up and eventually surpass test
accuracy. The point at which it crosses over and starts to pull away
is the point at which you'd consider stopping training or strengthening
regularisation. Watching the gap is a basic diagnostic skill.

---

## 6. Running it

You'll need the four uncompressed MNIST IDX files. The download script
in `data/` fetches them from a reliable mirror:

```
chmod +x data/download_mnist.sh
./data/download_mnist.sh
```

Then from `build/`:

```
make
./bin/examples/mnist_classifier              # uses ../data/mnist
./bin/examples/mnist_classifier /some/path   # or pass a path
```

Expected output: ~97% test accuracy after ten epochs, in a few minutes
of CPU time on a modern laptop.

---

## 7. What's missing (worth knowing)

A handful of things a "serious" MNIST classifier would do that we
deliberately don't:

- **Learning rate schedule.** Adam adapts per-parameter, but a global
  step-decay or cosine schedule on top can squeeze another half-percent
  of accuracy. Not worth the complexity here.
- **Data augmentation.** Random shifts of ±2 pixels and small rotations
  are the classic MNIST trick — they reliably take an MLP from ~98% to
  ~99%. Requires an image-aware augmentation layer we don't have.
- **Validation set.** We're using the test set for per-epoch evaluation,
  which is methodologically lax — in a published result you'd hold out
  a slice of training for validation and only touch the test set once
  at the end. For a learning project that's pedantry.
- **Model checkpointing.** Save the best-test-accuracy model rather
  than the last-epoch one. Easy to add when there's a reason to.
- **ConvNets.** The big jump from ~98% MLP to ~99.5%+ comes from
  exploiting the 2D structure of the image with convolutional layers
  instead of flattening to 784. That's a roadmap item — needs 4D
  tensors, Conv2D, and MaxPool.

The accuracy ceiling for a well-tuned MLP on MNIST is around 98-98.5%.
The state of the art (deep ConvNets with augmentation and ensembling)
sits at 99.8%, where the remaining errors are images even humans
disagree about.
