# 05 — Network: layers composed

By now, all the math in weft lives inside the individual layer and loss
classes — `Dense`, `ReLU`, `Softmax`, `CrossEntropy`. This note covers the
small class that composes them into a trainable model.

`Network` is the simplest class in the library, and it should be — its job
is just to chain things together.

---

## 1. Forward = function composition

A neural network is literally a composition of functions:

```
f  =  f_N  ∘  f_{N-1}  ∘  …  ∘  f_2  ∘  f_1
```

In code, that's a `for` loop that threads the output of one layer into the
input of the next:

```cpp
Matrix<T> A = X;
for (auto& layer : layers_)
    A = layer->forward(A);
return A;
```

That's the whole forward pass.

---

## 2. Backward = composition, reversed

The chain rule for composed functions is:

```
(f_N ∘ … ∘ f_1)'(X)  =  J_N  ·  J_{N-1}  ·  …  ·  J_1
```

where `J_k` is the Jacobian of layer `k`. From the `Jᵀ · dY` view in note
`03b_jacobian.md`, computing `∂L/∂X` from `∂L/∂Y` means *transposing* this
product and applying it to the upstream gradient — which is exactly
multiplying by `J_1ᵀ`, then by `J_2ᵀ`, … , then by `J_Nᵀ`. So we walk the
list in reverse:

```cpp
Matrix<T> g = dY;
for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
    g = (*it)->backward(g);
return g;
```

The reversal isn't a convention — it's the chain rule applied to a
composition, written out.

---

## 3. Ownership: `unique_ptr<Layer<T>>`

Each layer is heap-allocated, owned by a `std::unique_ptr<Layer<T>>`. The
Network keeps a `std::vector` of those.

Why heap and pointers? Polymorphism. `Dense<T>`, `ReLU<T>`, and
`Softmax<T>` are different concrete types but share the abstract base
`Layer<T>`. Storing them by value in a `vector<Layer<T>>` would slice them
(only the base part would be kept). Pointers preserve the dynamic type so
`layer->forward(A)` dispatches to the correct override.

`unique_ptr` (rather than raw pointers) makes ownership unambiguous and
automatic — when the Network is destroyed, every layer is destroyed with
it. No leaks, no manual `delete`.

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

Three things this gives us at the call site:

```cpp
net.add<Dense>(4, 16);   // construct Dense<float>(4, 16), add it
net.add<ReLU>();         // no args
auto& d = net.add<Dense>(16, 3);     // also returns a ref to the concrete layer,
d.W() = ...;                         // so we can configure it directly
```

The template-template parameter `template <typename> class L` lets the
user pass `Dense`, `ReLU`, etc. (the *templates*, unspecialised) and weft
fills in `L<T>` itself. The variadic `Args` forward to the chosen layer's
constructor. Returning a reference to the concrete type (not `Layer<T>&`)
lets tests and configuration code reach the layer's specific accessors
without `dynamic_cast`.

---

## 5. Loss stays outside Network

Some libraries treat the loss as a final layer. We don't, for two reasons.

- **Type mismatch**: layers map matrices to matrices; losses map matrices
  to a scalar. Forcing a single `forward` signature would be ugly.
- **Conceptual separation**: a Network produces *predictions*. A Loss
  compares them to targets and produces a gradient to start backprop with.
  Keeping them separate makes the training loop read like the math:

```cpp
Matrix<T> pred = net.forward(X);     // network produces predictions
T         L    = loss.forward(pred, target);   // loss measures them
Matrix<T> dPred = loss.backward();              // loss starts the gradient
net.backward(dPred);                            // network finishes it
net.update(lr);                                 // SGD step
```

This is also why PyTorch, TensorFlow, JAX, etc. all keep losses separate
from the model.

---

## 6. The full-pipeline numerical gradient check

The test that matters most is in `tests/test_network.cpp`: build the
complete classifier

```
Dense(3,5) → ReLU → Dense(5,4) → Softmax → CrossEntropy
```

with random weights and 4 examples of one-hot targets. Then for each
entry `W[i,j]` of the first Dense's weight matrix, compare:

- the analytical gradient produced by `net.backward(ce.backward())`, and
- the finite difference `(L(W + ε) − L(W − ε)) / (2 ε)`.

If they agree everywhere — and they do — then **every component in the
library is correctly wired together and computing the right gradient**.
Any bug in any layer's `backward`, in the Network's reverse iteration, in
CrossEntropy's `−T/S`, or in Softmax's Jacobian formula would show up here.

This is the strongest correctness test we can write short of training a
real model. Whenever you add a new layer to weft, the first thing to do
is drop it into the middle of a network like this and re-run the
numerical gradient check.

---

## 7. What we built

`src/Network.h` — owns layers via `unique_ptr<Layer<T>>`, provides
`forward`, `backward`, `update`, and the `add<L>` template helper.

`tests/test_network.cpp` — three tests:

1. Forward composition matches a hand-chained sequence (sanity check).
2. **Full-pipeline numerical gradient check** (the important one).
3. End-to-end SGD: a small 4-feature 3-class synthetic problem trains
   for 200 steps and cross-entropy loss drops by more than 3×. First
   time the library does real learning.

---

## 8. What's next: IRIS

We have every piece needed for a real classifier:

- `Matrix` and the operations on it.
- `Dense` (forward, backward, SGD update).
- `ReLU`, `Softmax` activations.
- `CrossEntropy` loss.
- `Network` to wire them.

The next step is the IRIS dataset: load it, normalise the features, split
into train/test, write a training loop, watch the loss curve, report
accuracy. That goes in `examples/iris_classifier/`.

After IRIS we'll also start collecting the roadmap items the project will
need as it scales:

- **Optimiser refactor + Adam**: factor `update(lr)` out of `Layer` into
  a separate `Optimizer` that holds per-parameter state.
- **Train/eval mode + Dropout**: layers gain a mode flag; Dropout drops a
  random fraction of activations during training, scales them up at
  inference (or uses inverted dropout).
- **Conv2D and MaxPool**: needs 4D tensors. Arrives with the ConvNet
  milestone.
