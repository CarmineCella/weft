#pragma once
//
// Softmax.h  --  Softmax activation, used as the final layer for multi-
//                class classification.  Per-example (per-column) operation;
//                no parameters.
//
//   forward (numerically stable):
//      m_j      = max over rows of Z[:, j]                     (per column)
//      S[i,j]   = exp(Z[i,j] - m_j) / sum_k exp(Z[k,j] - m_j)
//      ( softmax(z) == softmax(z - c)  for any c, so the shift by the max
//        does not change the result but keeps exp() from overflowing. )
//
//   backward (Jacobian of softmax):
//      For one example (one column),
//         ds_i / dz_j  =  s_i * (delta_ij - s_j)
//      Chain rule with dS = dL/dS gives
//         dZ[i,j]  =  S[i,j] * (dS[i,j] - dot_j)
//      where  dot_j = sum_k S[k,j] * dS[k,j]   (one scalar per column).
//
// Note: in classification networks Softmax is normally paired with cross-
// entropy loss; the *combined* gradient is just (S - target), much simpler
// than the standalone backward.  The Loss class will use that shortcut.
// This standalone backward is correct on its own and verified by the
// numerical gradient check in tests/test_softmax.cpp.
//
#include "Layer.h"
#include "Matrix.h"

#include <cmath>
#include <cstddef>

namespace weft {

template <typename T = float>
class Softmax : public Layer<T> {
public:
    Matrix<T> forward(const Matrix<T>& Z) override {
        const std::size_t R = Z.rows();
        const std::size_t C = Z.cols();
        Matrix<T> S(R, C);

        for (std::size_t j = 0; j < C; ++j) {
            // 1. column max -- the log-sum-exp shift for numerical stability.
            T m = Z(0, j);
            for (std::size_t i = 1; i < R; ++i)
                if (Z(i, j) > m) m = Z(i, j);

            // 2. exponentiate shifted entries; accumulate the denominator.
            T denom = T(0);
            for (std::size_t i = 0; i < R; ++i) {
                S(i, j) = std::exp(Z(i, j) - m);
                denom  += S(i, j);
            }

            // 3. normalise to a probability distribution.
            for (std::size_t i = 0; i < R; ++i)
                S(i, j) /= denom;
        }

        S_cache_ = S;     // backward needs S, not Z
        return S;
    }

    Matrix<T> backward(const Matrix<T>& dS) override {
        const std::size_t R = S_cache_.rows();
        const std::size_t C = S_cache_.cols();
        Matrix<T> dZ(R, C);

        for (std::size_t j = 0; j < C; ++j) {
            // dot_j = sum_i S[i,j] * dS[i,j]
            T dot = T(0);
            for (std::size_t i = 0; i < R; ++i)
                dot += S_cache_(i, j) * dS(i, j);

            // dZ[i,j] = S[i,j] * (dS[i,j] - dot_j)
            for (std::size_t i = 0; i < R; ++i)
                dZ(i, j) = S_cache_(i, j) * (dS(i, j) - dot);
        }
        return dZ;
    }
    // update() inherits the no-op default from Layer -- no parameters.

private:
    Matrix<T> S_cache_;   // forward output, needed by backward
};

} // namespace weft
