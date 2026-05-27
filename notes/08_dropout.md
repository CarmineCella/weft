# 08 — Train/eval mode and Dropout

Two changes that go together. We added a training/inference mode flag to
`Layer`, and we built `Dropout` — the first layer that actually uses it.

---

## 1. Why a mode flag exists at all

Most layers compute the same thing during training and during inference.
`Dense` is `WX + b` either way. `ReLU` is `max(0, X)` either way. The
loss function only exists during training, so it doesn't need to care.

But a small family of layers genuinely behaves differently in the two
phases:

- **Dropout** randomly zeroes activations during training, and is the
  identity at inference time.
- **Batch normalisation** uses the current batch's mean and variance
  during training, but uses *running averages* accumulated over training
  during inference.
- Some data-augmentation layers (random crops, random noise) similarly
  only fire during training.

Rather than have each of these layers carry their own flag and force
callers to wire it up, the flag lives in `Layer` and propagates from
`Network`:

```cpp
class Layer<T> {
public:
    virtual void set_training(bool t) { training_ = t; }
    bool training() const { return training_; }
private:
    bool training_ = true;
};
```

`Network` exposes `train()` and `eval()` that fan the flag out across
all layers. The default is training mode, because that's how every
freshly-constructed network is used — you only ever flip to eval
explicitly, on a validation or test set:

```cpp
net.train();
for (epoch = 0; ...) { ... training loop ... }

net.eval();
auto preds = net.forward(X_test);
```

The shape of this API is borrowed from PyTorch (`model.train()`,
`model.eval()`) deliberately. Muscle memory transfers.

---

## 2. Dropout — what and why

Dropout is a regularisation technique introduced by Srivastava et al.
(2014). The mechanism is brutally simple:

> During training, for each activation, with probability `p` set it to
> zero; otherwise scale it by `1/(1-p)`. During inference, do nothing.

The "scale it by `1/(1-p)`" trick is called **inverted dropout**, and
it's the reason inference can be a no-op. The expected value of the
output equals the input:

```
E[y]  =  p · 0  +  (1-p) · x / (1-p)  =  x
```

so on average the network sees activations with the same mean during
training and during inference. Without the scaling we'd have to scale
*down* at inference time instead (multiply activations by `1-p`), which
is awkward to remember and easy to forget. Pushing the math into the
training path keeps inference clean.

### Why it works

The intuition I find most compelling is **implicit ensembling**.

Each training forward pass with a different random mask is effectively
training a different sub-network — a randomly chosen subset of the
neurons. Over many batches we're really training an enormous family
of sub-networks that share weights. At inference time the full network
(no dropout) behaves approximately like an average over all those
sub-networks, and ensemble averages are reliably better than any single
member.

A second framing that's sometimes useful: dropout prevents **co-
adaptation**. Without dropout, two neurons in the same layer can settle
into a brittle "I will detect X *only when my neighbour detects Y*"
arrangement. Dropout breaks these dependencies — your neighbour might
not be there next batch — and forces each neuron to be individually
useful.

Both stories point at the same outcome: a network that overfits less
and generalises better, especially when the model has many more
parameters than the data really needs.

### Common rates

- `0.5` is the classic Srivastava setting, used on fully-connected
  layers in big networks.
- `0.2`-`0.3` is more common in modern practice.
- `0.1` or less for layers close to the input.
- ConvNets typically dropout less than fully-connected nets (and use
  spatial dropout variants instead — out of scope here).

### Where to place it

After a Dense layer, typically after its activation. The pattern is

```
Dense -> ReLU -> Dropout -> Dense -> ReLU -> Dropout -> Dense -> Softmax
                                                                  ^
                                                  no dropout on the
                                                  classification head
```

You don't put dropout on the output of a softmax classifier — you'd be
zeroing class probabilities, which is nonsense. You also rarely put
dropout right before the final Dense.

---

## 3. The code

```cpp
template <typename T = float>
class Dropout : public Layer<T> {
public:
    explicit Dropout(T rate, unsigned seed = std::random_device{}())
        : rate_(rate), gen_(seed) {}

    Matrix<T> forward(const Matrix<T>& X) override {
        if (!this->training() || rate_ == T(0))
            return X;

        const T scale = T(1) / (T(1) - rate_);
        std::bernoulli_distribution keep(double(T(1) - rate_));
        mask_ = Matrix<T>(X.rows(), X.cols());
        for (size_t i = 0; i < X.rows(); ++i)
            for (size_t j = 0; j < X.cols(); ++j)
                mask_(i, j) = keep(gen_) ? scale : T(0);

        return hadamard(X, mask_);
    }

    Matrix<T> backward(const Matrix<T>& dY) override {
        if (!this->training() || rate_ == T(0))
            return dY;
        return hadamard(dY, mask_);
    }

private:
    T rate_;
    std::mt19937 gen_;
    Matrix<T> mask_;
};
```

A few things worth noting:

The mask is stored on the layer, exactly like `ReLU` stores its
`X > 0` mask. The two layers' backward methods are structurally
identical — both compute `dY ⊙ mask`. The only difference is how the
mask is built.

The random number generator is seeded from `std::random_device` by
default, but the constructor accepts an explicit seed, which is what
the tests use to get reproducible behaviour. In a real training run
you almost certainly want fresh randomness every time, so the default
is the right one.

The two early-return paths — eval mode and `rate == 0` — short-circuit
to the identity. `rate == 0` is admittedly a degenerate case, but it
lets you wire a `Dropout` layer into a network without committing to
using it, and toggle it on later by setting `rate_` non-zero. (We
don't expose a setter for this yet, but the point stands.)

### Testing dropout

We deliberately don't do a numerical gradient check on Dropout, because
the forward pass is stochastic — running `forward(X + ε e_ij)` and
`forward(X - ε e_ij)` generates *two different masks*, and the
gradient check becomes meaningless.

Instead we test the algebraic identity directly. With `X` all ones, the
output `Y` equals the mask (`Y = X ⊙ M = M`), so the backward output
satisfies `dX(i,j) = M(i,j) * dY(i,j) = Y(i,j) * dY(i,j)`. We check
that elementwise. We also verify the scaling-preserves-the-mean
property over a 10,000-element matrix, and that `Network::train()` and
`Network::eval()` actually propagate the flag down to a Dropout layer
embedded in the network.

---

## 4. Looking ahead

With dropout in place we can sensibly train a network that has many
more parameters than the data really supports — which is exactly the
situation on MNIST (60k examples, but our networks will have hundreds
of thousands of weights). Combined with Adam, this is enough machinery
for a respectable MNIST classifier.

The MNIST loader is in the same release as this note; it's a tiny IDX
parser that returns a `(784, N)` matrix of pixels in `[0, 1]` and a
`std::vector<int>` of labels. The actual MNIST classifier example —
network architecture, training loop with eval-mode test passes, results
— is the next step.
