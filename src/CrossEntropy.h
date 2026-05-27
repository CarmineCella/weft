#pragma once
//
// CrossEntropy.h  --  categorical cross-entropy loss for classification.
//
//   forward (mean over batch):
//      L  =  -(1/N)  *  sum_n sum_i  T[i,n] * log(S[i,n])
//
//      where S is the predicted distribution (shape: classes x batch),
//      T is the target distribution (typically one-hot, same shape).
//
//   backward (returned as dL/dS):
//      dL/dS[i,n]  =  -(1/N) * T[i,n] / S[i,n]
//
// Pair this with Softmax: when CE.backward()'s output is fed through
// Softmax::backward(), the chain simplifies algebraically to
//      dL/dZ  =  (S - T) / N
// (the standalone formulas individually contain things that can blow up
// when S has tiny entries; the composition has those terms cancel out).
// That identity is verified in tests/test_crossentropy.cpp.
//
#include "Loss.h"
#include "Matrix.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace weft {

template <typename T = float>
class CrossEntropy : public Loss<T> {
public:
    T forward(const Matrix<T>& predicted, const Matrix<T>& target) override {
        if (predicted.rows() != target.rows() ||
            predicted.cols() != target.cols())
            throw std::invalid_argument(
                "CrossEntropy::forward: shape mismatch between predicted and target");

        predicted_cache_ = predicted;
        target_cache_    = target;

        const T eps = T(1e-12);     // guard against log(0); harmless on healthy values
        T total = T(0);
        for (std::size_t j = 0; j < predicted.cols(); ++j)
            for (std::size_t i = 0; i < predicted.rows(); ++i)
                total -= target(i, j) * std::log(predicted(i, j) + eps);

        return total / static_cast<T>(predicted.cols());
    }

    Matrix<T> backward() override {
        const std::size_t R = predicted_cache_.rows();
        const std::size_t N = predicted_cache_.cols();
        const T eps   = T(1e-12);
        const T inv_N = T(1) / static_cast<T>(N);

        Matrix<T> dS(R, N);
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < R; ++i)
                dS(i, j) = -inv_N * target_cache_(i, j) /
                           (predicted_cache_(i, j) + eps);
        return dS;
    }

private:
    Matrix<T> predicted_cache_;
    Matrix<T> target_cache_;
};

} // namespace weft
