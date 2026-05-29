#pragma once
//
// ConvLayer.h  --  abstract base class for layers operating on Tensor4D.
//
// Conv layers fundamentally take 4-D tensors (batch, channels, height,
// width) in and produce 4-D tensors out, so they need a different
// forward/backward signature from the dense Layer class.  Rather than
// trying to unify the two under a single base class (which forces type
// erasure or template wizardry that obscures the data flow), this is a
// separate, parallel hierarchy.  The data-flow bridge between the two
// is a free function `flatten(Tensor4D) -> Matrix`, used in the example
// once the conv stack has finished and the dense classifier starts.
//
// The interface mirrors Layer<T> on purpose: forward/backward/update,
// a training flag, a describe() string for net.summary(), and optional
// save_params / load_params for serialization.  Anyone who has read
// Layer.h already knows how this works.
//
#include "Tensor4D.h"

#include <string>
#include <iosfwd>

namespace weft {

template <typename T> class Optimizer;   // fwd decl

template <typename T = float>
class ConvLayer {
public:
    virtual ~ConvLayer() = default;

    // Forward and backward operate on Tensor4D.  Same contract as Layer:
    //   forward(X)  caches whatever the layer needs for its backward,
    //   backward(dY) returns the upstream gradient dX and stashes any
    //   parameter gradients (dW, db, ...) on the layer object,
    //   update(opt) hands those parameter gradients to the optimizer.
    virtual Tensor4D<T> forward (const Tensor4D<T>& X)  = 0;
    virtual Tensor4D<T> backward(const Tensor4D<T>& dY) = 0;
    virtual void        update (Optimizer<T>& opt)      = 0;

    void set_training(bool t) { training_ = t; }
    bool training() const     { return training_; }

    // For ConvNetwork::summary() / debugging.
    virtual std::string describe() const = 0;

    // For ConvNetwork::save() / load().  Default no-op suits parameter-
    // free layers (ReLU4D, MaxPool2D); Conv2D overrides them.
    virtual void save_params(std::ostream&)       const {}
    virtual void load_params(std::istream&)             {}

protected:
    bool training_ = true;
};

} // namespace weft
