# 8 — Network: layers composed

All the math in weft is inside the individual layer and loss classes:
`Dense`, `ReLU`, `Softmax`, `CrossEntropy`. This chapter covers the
small class that composes them into a trainable model.

`Network` is the simplest class in the library. Its only job is to call
the layers in sequence.

---

## 1. Forward = function composition

A neural network is a composition of functions:

```
f  =  f_N  ∘  f_{N-1}  ∘  …  ∘  f_2  ∘  f_1
```

In code, the whole forward pass is a `for` loop that passes the output
of one layer to the input of the next:

```cpp
Matrix<T> A = X;
for (auto& layer : layers_)
    A = layer->forward(A);
return A;
```

---

## 2. Backward = composition, reversed

The chain rule for composed functions is:

```
(f_N ∘ … ∘ f_1)'(X)  =  J_N  ·  J_{N-1}  ·  …  ·  J_1
```

where `J_k` is the Jacobian of layer `k`. From the `Jᵀ · dY` view in
[chapter 6](06_jacobian.md), computing `∂L/∂X` from `∂L/∂Y` means
transposing this product and applying it to the upstream gradient. That
is the same as multiplying `∂L/∂Y` by `J_Nᵀ` first, then by
`J_{N-1}ᵀ`, … , and finally by `J_1ᵀ`. So we traverse the list in
reverse:

```cpp
Matrix<T> g = dY;
for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
    g = (*it)->backward(g);
return g;
```

The reverse order follows from the chain rule applied to a composition;
it is not a convention.

---

## 3. Ownership: `unique_ptr<Layer<T>>`

Each layer is heap-allocated, owned by a `std::unique_ptr<Layer<T>>`. The
Network keeps a `std::vector` of those.

The heap and the pointers are needed for polymorphism. `Dense<T>`,
`ReLU<T>`, and `Softmax<T>` are different concrete types but share the
abstract base `Layer<T>`. Storing them by value in a `vector<Layer<T>>`
would slice them (only the base part would be kept). Pointers preserve
the dynamic type so `layer->forward(A)` dispatches to the correct
override.

`unique_ptr` (rather than raw pointers) makes ownership unambiguous and
automatic: when the Network is destroyed, every layer is destroyed with
it. There are no leaks and no manual `delete`.

---

## 4. The `add<L>(args...)` template helper

```cpp
template <template <typename> class L, typename... Args>
L<T>& add(Args&&... args) {
    auto ptr = std::make_unique<L<T>>(std::forward<Args>(args)...);
    L<T>* raw = ptr.get();
    layers_.push_back(std::move(ptr));
    return *raw;
}
```

At the call site this gives:

```cpp
net.add<Dense>(4, 16);   // construct Dense<float>(4, 16), add it
net.add<ReLU>();         // no args
auto& d = net.add<Dense>(16, 3);     // also returns a ref to the concrete layer,
d.W() = ...;                         // so we can configure it directly
```

The template-template parameter `template <typename> class L` lets the
user pass `Dense`, `ReLU`, etc. (the unspecialised templates) and weft
fills in `L<T>` itself. The variadic `Args` forward to the chosen layer's
constructor. Returning a reference to the concrete type (not `Layer<T>&`)
lets tests and configuration code reach the layer's specific accessors
without `dynamic_cast`.

---

## 5. Loss stays outside Network

Some libraries treat the loss as a final layer. We do not.

- **Type mismatch**: layers map matrices to matrices; losses map matrices
  to a scalar. A single `forward` signature for both would be awkward.
- **Conceptual separation**: a Network produces predictions. A Loss
  compares them to targets and produces the gradient that starts
  backprop. Keeping them separate makes the training loop follow the
  math:

```cpp
Matrix<T> pred = net.forward(X);                // network produces predictions
T         L    = loss.forward(pred, target);    // loss measures them
Matrix<T> dPred = loss.backward();              // loss starts the gradient
net.backward(dPred);                            // network finishes it
net.update(opt);                                // optimiser applies the step
```

PyTorch, TensorFlow, JAX and others keep losses separate from the model
for the same reasons.

---

## 6. The full-pipeline numerical gradient check

The most important test is in `tests/test_network.cpp`. It builds the
complete classifier

```
Dense(3,5) → ReLU → Dense(5,4) → Softmax → CrossEntropy
```

with random weights and 4 examples of one-hot targets. Then for each
entry `W[i,j]` of the first Dense's weight matrix, it compares:

- the analytical gradient produced by `net.backward(ce.backward())`, and
- the finite difference `(L(W + ε) − L(W − ε)) / (2 ε)`.

They agree everywhere, which shows that every component in the library
is correctly connected and computes the right gradient. A bug in any
layer's `backward`, in the Network's reverse iteration, in
CrossEntropy's `−T/S`, or in Softmax's Jacobian formula would show up
here.

This is the strongest correctness test we can write short of training a
real model. When you add a new layer to weft, first put it in the middle
of a network like this one and re-run the numerical gradient check.

---

## 7. What we built {#sec-08-network-what-we-built}

The network owns its layers through `unique_ptr`. Its interface provides
`forward`, `backward`, `update`, and the `add<L>` template helper (plus
`train()`/`eval()`, `summary()` and `save`/`load`, used from chapter 11
on).

`tests/test_network.cpp` has seven checks:

1. Forward composition matches a hand-chained sequence (sanity check).
2. The full-pipeline numerical gradient check described above, the most
   important one.
3. End-to-end SGD: a small 4-feature 3-class synthetic problem trains
   for 200 steps and cross-entropy loss drops by more than 3×. It is
   the first training run in the library.
4. `summary()` and `describe()` print the architecture.
5. `save` then `load` round-trips the parameters, and loading rejects
   mismatched layer counts or parameter shapes. These checks do not
   distinguish every same-shaped architecture; chapter 19 describes
   the serialization contract.

---

## 8. The components so far

The library now has every component needed for a classifier:

- `Matrix` and the operations on it.
- `Dense` (forward, backward, SGD update).
- `ReLU`, `Softmax` activations.
- `CrossEntropy` loss.
- `Network` to compose them.

The next chapter applies them to the IRIS dataset: it loads the data,
normalises the features, splits into train/test, runs a training loop,
follows the loss curve and reports accuracy. The code is in
`examples/iris_classifier.cpp`.

Later chapters add:

- **Train/eval mode and Dropout** ([chapter 11](11_dropout.md)): layers
  gain a mode flag; Dropout zeroes a random fraction of activations during
  training and scales the survivors by `1/(1−p)` (inverted dropout), so
  that inference is the identity.
- **Conv2D and MaxPool** (chapters [21](21_from_matrices_to_tensors.md)
  and [22](22_convnets.md)): these need 4-D tensors.


## Further reading {#sec-08-network-further-reading}

Building a network as a chain of modules, each with its own forward and
backward function, is the design described in [@lecun1998gradient].
