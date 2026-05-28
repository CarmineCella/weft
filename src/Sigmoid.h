#pragma once
//
// Sigmoid.h  --  logistic activation, sigma(x) = 1 / (1 + e^-x).
//
//   forward:   Y[i,j] = sigma(Z[i,j])                  (element-wise)
//   backward:  dZ = dY elementwise-times Y (1 - Y)
//
// The derivative of sigma is sigma * (1 - sigma), and since the forward
// pass already computed Y = sigma(Z), backward reuses Y directly rather
// than recomputing the exponential.  Like ReLU, the Jacobian is diagonal
// (element-wise activation), so backward is a Hadamard product.
//
// Squashes any real input into (0, 1).  We use it as the output
// activation of an autoencoder reconstructing values in [0, 1] (e.g.
// normalised pixel intensities), so the reconstruction can't stray
// outside the valid range.  Also the classic gate nonlinearity for
// future recurrent layers.
//
// No parameters; preserves input shape.
//
#include "Layer.h"
#include "Matrix.h"

#include <cmath>

namespace weft {

template <typename T = float>
class Sigmoid : public Layer<T> {
public:
    Matrix<T> forward(const Matrix<T>& Z) override {
        Y_ = Z.apply([](T x){ return T(1) / (T(1) + std::exp(-x)); });
        return Y_;
    }

    Matrix<T> backward(const Matrix<T>& dY) override {
        // dZ = dY * Y * (1 - Y), element-wise.
        Matrix<T> deriv = Y_.apply([](T y){ return y * (T(1) - y); });
        return hadamard(dY, deriv);
    }
    // update() inherits the no-op default from Layer -- no parameters.

private:
    Matrix<T> Y_;   // cached output sigma(Z), reused in backward
};

} // namespace weft
