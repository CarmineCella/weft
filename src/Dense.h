#pragma once
//
// Dense.h  --  a fully connected (a.k.a. linear, a.k.a. affine) layer.
//
//   forward:   Z = W * X + b
//   backward:  given dL/dZ,
//                  dW = dZ * X^T
//                  db = dZ summed across the batch dimension
//                  dX = W^T * dZ      (returned to the layer below)
//
// Shapes (weft column convention -- one example per column):
//     W       (out_features  x  in_features)
//     b       (out_features  x  1)
//     X       (in_features   x  batch_size)
//     Z = W*X + b   (out_features  x  batch_size)
//
#include "Layer.h"
#include "Matrix.h"

#include <cmath>
#include <cstddef>

namespace weft {

template <typename T = float>
class Dense : public Layer<T> {
public:
    Dense(std::size_t in_features, std::size_t out_features)
        : W_(out_features, in_features),
          b_(out_features, 1, T(0))
    {
        // He initialisation: stddev = sqrt(2 / fan_in).
        // Keeps activation variances roughly constant across layers when
        // paired with ReLU (we'll dig into this when we add ReLU).
        const T stddev = std::sqrt(T(2) / static_cast<T>(in_features));
        W_.randomizeNormal(T(0), stddev);
    }

    // ------------------------------------------------------------------
    // forward:  Z = W * X + b
    // We cache X because backward needs it for dW.
    // ------------------------------------------------------------------
    Matrix<T> forward(const Matrix<T>& X) override {
        X_cache_ = X;
        return W_ * X + b_;
    }

    // ------------------------------------------------------------------
    // backward:
    //   dW = dZ * X^T             (sum over the batch happens inside the matmul)
    //   db = dZ.sumColumns()      (explicit sum over the batch)
    //   dX = W^T * dZ             (sent upstream)
    // ------------------------------------------------------------------
    Matrix<T> backward(const Matrix<T>& dZ) override {
        dW_ = dZ * X_cache_.transpose();
        db_ = dZ.sumColumns();
        return W_.transpose() * dZ;
    }

    // Hand each parameter to the optimiser.  The optimiser knows whether
    // it's plain SGD, Adam, or anything else; the layer just exposes
    // (parameter, gradient) pairs.
    void update(Optimizer<T>& opt) override {
        opt.step(W_, dW_);
        opt.step(b_, db_);
    }

    // ------- accessors (used by tests; later, by the optimizer) -------
    Matrix<T>&       W()        { return W_; }
    const Matrix<T>& W()  const { return W_; }
    Matrix<T>&       b()        { return b_; }
    const Matrix<T>& b()  const { return b_; }
    const Matrix<T>& dW() const { return dW_; }
    const Matrix<T>& db() const { return db_; }

private:
    Matrix<T> W_, b_;        // parameters
    Matrix<T> dW_, db_;      // gradients (set by backward)
    Matrix<T> X_cache_;      // input cached during forward
};

} // namespace weft
