#pragma once
//
// ScaledDotProductAttention.h  --  the core attention operator.
//
// Given three matrices,
//
//   Q  (d_k x n_q)    queries
//   K  (d_k x n_k)    keys
//   V  (d_v x n_k)    values
//
// computes the attention-weighted output
//
//   out  (d_v x n_q)  =  V * softmax((K^T Q) / sqrt(d_k))
//                                    \---- (n_k x n_q) ----/
//
// Each column i of out is a weighted combination of the value columns,
// the weights being the softmax-normalised similarities of query i to
// each key.  Softmax is applied per column (i.e., for each query the
// attention weights over the keys sum to 1).
//
// No learnable parameters live here; the Q/K/V projections sit one level
// up in MultiHeadAttention, which lets a single MHA layer share this
// operator across its heads.  We keep it as a plain struct (not a
// Layer<T>) because Layer's forward(X) -> Y interface only carries one
// input, while attention has three.  MultiHeadAttention drives the
// caching and the gradient routing.
//
// Backward derivation.  Let S = (K^T Q) / sqrt(d_k) and A = softmax(S, axis=0)
// so out = V A.  Given dOut := dL/dOut:
//
//   dV = dOut A^T                                              (d_v x n_k)
//   dA = V^T dOut                                              (n_k x n_q)
//   dS = softmax_backward(dA, A)   per column,
//        dS[k,i] = A[k,i] * (dA[k,i] - sum_j A[j,i] * dA[j,i])
//   dQ = (K  @ dS  ) / sqrt(d_k)                               (d_k x n_q)
//   dK = (Q  @ dS^T) / sqrt(d_k)                               (d_k x n_k)
//
// (The Softmax class already implements its column-wise backward; we
// just feed dA in.)  Verified by finite-difference test in
// tests/test_scaled_dot_product_attention.cpp.
//
#include "Matrix.h"
#include "Softmax.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace weft {

template <typename T = float>
class ScaledDotProductAttention {
public:
    // Q (d_k x n_q), K (d_k x n_k), V (d_v x n_k)  ->  out (d_v x n_q)
    Matrix<T> forward(const Matrix<T>& Q,
                      const Matrix<T>& K,
                      const Matrix<T>& V)
    {
        if (Q.rows() != K.rows())
            throw std::invalid_argument(
                "ScaledDotProductAttention::forward: Q and K must share d_k");
        if (K.cols() != V.cols())
            throw std::invalid_argument(
                "ScaledDotProductAttention::forward: K and V must share n_k");

        Q_ = Q; K_ = K; V_ = V;
        scale_ = T(1) / std::sqrt(static_cast<T>(Q.rows()));

        Matrix<T> S = K.transpose() * Q;         // (n_k x n_q)
        S = S * scale_;                          // scale BEFORE softmax
        A_ = softmax_.forward(S);                // per-column softmax (n_k x n_q)
        return V * A_;                           // (d_v x n_q)
    }

    // dOut (d_v x n_q)  ->  (dQ, dK, dV)
    //   dQ (d_k x n_q), dK (d_k x n_k), dV (d_v x n_k)
    std::tuple<Matrix<T>, Matrix<T>, Matrix<T>>
    backward(const Matrix<T>& dOut)
    {
        Matrix<T> dV = dOut * A_.transpose();       // (d_v x n_k)
        Matrix<T> dA = V_.transpose() * dOut;       // (n_k x n_q)
        Matrix<T> dS = softmax_.backward(dA);       // (n_k x n_q)
        Matrix<T> dQ = (K_ * dS) * scale_;          // (d_k x n_q)
        Matrix<T> dK = (Q_ * dS.transpose()) * scale_;  // (d_k x n_k)
        return std::make_tuple(dQ, dK, dV);
    }

    // Inspection helpers; convenient for tests and for visualising the
    // attention map afterwards.
    const Matrix<T>& attention() const { return A_; }
    T                scale()     const { return scale_; }

private:
    Matrix<T>  Q_, K_, V_;   // saved inputs for backward
    Matrix<T>  A_;           // saved attention weights for backward
    T          scale_ = T(1);
    Softmax<T> softmax_;     // owns its own S_cache_, independent per instance
};

} // namespace weft
