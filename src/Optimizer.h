#pragma once
//
// Optimizer.h  --  abstract base class for parameter-update strategies.
//
//   step(param, grad)  :  apply one update to `param` using `grad`.
//                         Stateful optimisers (Adam, RMSProp, ...) use the
//                         address of `param` as the identity key for their
//                         per-parameter state, with lazy allocation on
//                         first contact.
//
// A layer's update() just hands each parameter to the optimiser one at a
// time. The optimiser decides everything else -- plain SGD, momentum,
// per-parameter adaptive learning rates, etc.
//
#include "Matrix.h"

namespace weft {

template <typename T = float>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Both arguments must have the same shape.
    virtual void step(Matrix<T>& param, const Matrix<T>& grad) = 0;
};

} // namespace weft
