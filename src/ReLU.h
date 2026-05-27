#pragma once
//
// ReLU.h  --  Rectified Linear Unit activation.
//
//   forward:   A[i,j] = max(0, Z[i,j])                 (element-wise)
//   backward:  dZ = dA elementwise-times mask          (route through where Z>0)
//              where mask[i,j] = 1 if Z[i,j] > 0 else 0.
//
// Why ReLU is so simple in backprop: it is element-wise, so the Jacobian
// is diagonal -- every output depends only on the corresponding input.
// The backward pass collapses to a Hadamard product.
//
// No parameters; preserves input shape.
//
#include "Layer.h"
#include "Matrix.h"

namespace weft {

template <typename T = float>
class ReLU : public Layer<T> {
public:
    Matrix<T> forward(const Matrix<T>& Z) override {
        // Cache the mask "Z > 0" -- that is everything backward needs.
        mask_ = Z.apply([](T x){ return x > T(0) ? T(1) : T(0); });
        return Z.apply([](T x){ return x > T(0) ? x : T(0); });
    }

    Matrix<T> backward(const Matrix<T>& dA) override {
        // Gradient flows where the input was positive, blocked elsewhere.
        return hadamard(dA, mask_);
    }
    // update() inherits the no-op default from Layer -- no parameters.

private:
    Matrix<T> mask_;   // recorded during forward, used in backward
};

} // namespace weft
