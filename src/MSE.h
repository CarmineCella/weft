#pragma once
//
// MSE.h  --  mean squared error loss, for regression and autoencoders.
//
//   forward (mean over batch):
//      L  =  (1 / (2N))  *  sum_n sum_i  (P[i,n] - T[i,n])^2
//
//      where P is the prediction (shape: features x batch), T is the
//      target (same shape).  Summed over features (rows), averaged over
//      examples (columns).  The factor of 1/2 makes the gradient clean
//      (it cancels the 2 from differentiating the square).
//
//   backward (returned as dL/dP):
//      dL/dP[i,n]  =  (1/N) * (P[i,n] - T[i,n])
//
// As with CrossEntropy, the 1/N batch-averaging factor lives here in the
// gradient, so layer backward formulas never need to know the batch size.
//
// For an autoencoder, `target` is just the network's own input -- the
// network is trained to reproduce what it was given.
//
#include "Loss.h"
#include "Matrix.h"

#include <cstddef>
#include <stdexcept>

namespace weft {

template <typename T = float>
class MSE : public Loss<T> {
public:
    T forward(const Matrix<T>& predicted, const Matrix<T>& target) override {
        if (predicted.rows() != target.rows() ||
            predicted.cols() != target.cols())
            throw std::invalid_argument(
                "MSE::forward: shape mismatch between predicted and target");

        predicted_cache_ = predicted;
        target_cache_    = target;

        T total = T(0);
        for (std::size_t j = 0; j < predicted.cols(); ++j)
            for (std::size_t i = 0; i < predicted.rows(); ++i) {
                const T d = predicted(i, j) - target(i, j);
                total += d * d;
            }
        return total / (T(2) * static_cast<T>(predicted.cols()));
    }

    Matrix<T> backward() override {
        const std::size_t R = predicted_cache_.rows();
        const std::size_t N = predicted_cache_.cols();
        const T inv_N = T(1) / static_cast<T>(N);

        Matrix<T> dP(R, N);
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < R; ++i)
                dP(i, j) = inv_N * (predicted_cache_(i, j) - target_cache_(i, j));
        return dP;
    }

private:
    Matrix<T> predicted_cache_;
    Matrix<T> target_cache_;
};

} // namespace weft
