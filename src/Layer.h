#pragma once
//
// Layer.h  --  abstract base class for every layer in weft.
//
// A layer plays two roles:
//
//   forward(X)   :  X         ->  Y                 (compute output)
//   backward(dY) :  dL/dY     ->  dL/dX             (route gradient upstream)
//                                                   and, internally, compute
//                                                   dL/dW, dL/db, ... for any
//                                                   trainable parameters.
//
//   update(lr)   :  apply the gradients stored in backward() with a plain
//                   gradient-descent step.  Layers with no parameters (e.g.
//                   activations) just do nothing.
//
// All layers are templated on the scalar type T (float / double).
//
#include "Matrix.h"
#include "Optimizer.h"

#include <string>

namespace weft {

template <typename T = float>
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass.  Layers that need information from forward in backward
    // (e.g. Dense caches its input) should store it here.
    virtual Matrix<T> forward(const Matrix<T>& X) = 0;

    // Backward pass.  Receives the upstream gradient dL/dY and returns the
    // downstream gradient dL/dX, while storing any parameter gradients.
    virtual Matrix<T> backward(const Matrix<T>& dY) = 0;

    // One step of parameter update using the given optimiser.
    // Default is a no-op for parameter-less layers (activations).
    virtual void update(Optimizer<T>& /*opt*/) {}

    // Training-vs-inference mode.  Some layers (Dropout, later BatchNorm)
    // compute different things at training time than at inference time.
    // Defaults to training mode -- you only ever explicitly switch to eval.
    virtual void set_training(bool t) { training_ = t; }
    bool training() const { return training_; }

    // One-line, human-readable description of this layer, e.g.
    // "Dense(784, 128)", "ReLU", "Dropout(0.30)".  Network::summary()
    // builds the printed architecture from these, so the description is
    // generated from the actual layers rather than hand-written (and so
    // can never silently drift out of sync with the real network).
    virtual std::string describe() const = 0;

private:
    bool training_ = true;
};

} // namespace weft
