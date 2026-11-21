# 11 — Train/eval mode and Dropout

This chapter adds a training/inference mode flag to `Layer`, and
`Dropout`, the first layer that uses it.

---

## 1. Why a mode flag exists at all

Most layers compute the same thing during training and during inference.
`Dense` is `WX + b` either way. `ReLU` is `max(0, X)` either way. The
loss function only exists during training, so the distinction does not
apply to it.

A few layers behave differently in the two phases:

- **Dropout** randomly zeroes activations during training, and is the
  identity at inference time.
- **Batch normalisation** uses the current batch's mean and variance
  during training, but uses running averages accumulated over training
  during inference.
- Some data-augmentation layers (random crops, random noise) are
  likewise active only during training.

Instead of each of these layers having its own flag that callers must
set, the flag is in `Layer` and is propagated from `Network`:

```cpp
class Layer<T> {
public:
    virtual void set_training(bool t) { training_ = t; }
    bool training() const { return training_; }
private:
    bool training_ = true;
};
```

`Network` provides `train()` and `eval()`, which set the flag on all
layers. The default is training mode, because that is how a newly
constructed network is used. You switch to eval explicitly, on a
validation or test set:

```cpp
net.train();
for (epoch = 0; ...) { ... training loop ... }

net.eval();
auto preds = net.forward(X_test);
```

This API is deliberately copied from PyTorch (`model.train()`,
`model.eval()`), so that habits carry over.

---

## 2. Dropout — what and why

Dropout is a regularisation technique introduced by Srivastava et al.
(2014) [@srivastava2014dropout]. The mechanism is simple:

> During training, for each activation, with probability `p` set it to
> zero; otherwise scale it by `1/(1-p)`. During inference, do nothing.

The scaling by `1/(1-p)` is called **inverted dropout**, and it is the
reason inference can be a no-op. The expected value of the output equals
the input:

```
E[y]  =  p · 0  +  (1-p) · x / (1-p)  =  x
```

so on average the network sees activations with the same mean during
training and during inference. Without the scaling we would have to
scale down at inference time instead (multiply activations by `1-p`),
which is easy to forget. Doing the scaling in the training path leaves
inference unchanged.

### Why it works

The explanation we find most convincing is **implicit ensembling**.

Each training forward pass with a different random mask in effect
trains a different sub-network, made of a randomly chosen subset of the
neurons. Over many batches we train a very large family of sub-networks
that share weights. At inference time the full network (no dropout)
behaves approximately like an average over all those sub-networks, and
ensemble averages are reliably better than any single member.

A second explanation is that dropout prevents **co-adaptation**.
Without dropout, two neurons in the same layer can settle into a
fragile arrangement in which one detects X only when the other detects
Y. Dropout breaks these dependencies, because the other neuron may be
absent in the next batch, and forces each neuron to be useful on its
own.

Both explanations motivate dropout as a regularizer. Whether it improves
generalization for a particular model needs a validation comparison,
especially when changing model capacity or the amount of training data.

### Common rates

- `0.5` is the classic Srivastava setting, used on fully-connected
  layers in big networks.
- `0.2`-`0.3` is more common in modern practice.
- `0.1` or less for layers close to the input.
- ConvNets typically use less dropout than fully-connected nets (and
  use spatial dropout variants instead, which are out of scope here).

### Where to place it

After a Dense layer, typically after its activation. The pattern is

```
Dense -> ReLU -> Dropout -> Dense -> ReLU -> Dropout -> Dense -> Softmax
                                                                  ^
                                                  no dropout on the
                                                  classification head
```

Do not put dropout on the output of a softmax classifier: it would zero
class probabilities, which makes no sense. Dropout before the final
Dense, as in the diagram, is fine. Dropout after it is what to avoid.

---

## 3. The code

```cpp
template <typename T = float>
class Dropout : public Layer<T> {
public:
    explicit Dropout(T rate, unsigned seed = random_seed())
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

The mask is stored on the layer, as `ReLU` stores its `X > 0` mask. The
backward methods of the two layers have the same structure: both
compute `dY ⊙ mask`. The only difference is how the mask is built.

Each layer owns a random-number stream. Its default seed is drawn from
weft's master seed generator, and the constructor also accepts an
explicit seed. Call `weft::set_seed(42)` before building a model to
repeat its initialization and dropout streams on the same toolchain.
The example programs set a seed; different seeds are useful controlled
experiments rather than an accidental change on every run.

The two early-return paths, eval mode and `rate == 0`, return the
identity. `rate == 0` is a degenerate case, but it lets you put a
`Dropout` layer into a network without using it, and enable it later by
setting `rate_` non-zero. (There is no setter for this yet.)

### Testing dropout

We do not run a numerical gradient check on Dropout, because the
forward pass is stochastic: `forward(X + ε e_ij)` and
`forward(X - ε e_ij)` generate two different masks, and the gradient
check becomes meaningless.

Instead we test the algebraic identity directly. With `X` all ones, the
output `Y` equals the mask (`Y = X ⊙ M = M`), so the backward output
satisfies `dX(i,j) = M(i,j) * dY(i,j) = Y(i,j) * dY(i,j)`. We check
that elementwise. We also verify that the scaling preserves the mean
over a 10,000-element matrix, and that `Network::train()` and
`Network::eval()` propagate the flag to a Dropout layer inside a
network.

---

## 4. Use on MNIST

With dropout we can train a network that has many more parameters than
the data supports. This is the situation on MNIST (60k examples, and
networks with hundreds of thousands of weights). Together with Adam,
this is enough for a good MNIST classifier.

The MNIST loader (`src/MNIST.h`) is a small IDX
parser that returns a `(784, N)` matrix of pixels in `[0, 1]` and a
`std::vector<int>` of labels. The MNIST classifier example (network
architecture, training loop with eval-mode test passes, results) is the
subject of the next chapter.


## Further reading {#sec-11-dropout-further-reading}

Dropout is [@srivastava2014dropout]. Batch normalisation, mentioned here
as the other layer that behaves differently in training and inference,
is [@ioffe2015batchnorm].
