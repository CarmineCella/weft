#pragma once
//
// Flatten.h  --  the bridge between Tensor4D (conv layers) and Matrix
//                (dense layers).  Free functions, not a Layer class,
//                because the conversion is just a reshape -- there is
//                no learnable state and no caching needed (the example
//                holds the input shape itself to call unflatten() in
//                the backward pass).
//
// In a classic conv classifier (AlexNet-shape):
//
//     Tensor4D<T> features = conv.forward(X);         // (N, C, H, W)
//     Matrix<T>   flat     = flatten(features);       // (C*H*W, N)
//     Matrix<T>   logits   = dense.forward(flat);     // (10, N)   etc.
//
// And in the matching backward pass:
//
//     Matrix<T>   dFlat    = dense.backward(dLogits); // (C*H*W, N)
//     Tensor4D<T> dFeat    = unflatten(dFlat, N, C, H, W);
//     conv.backward(dFeat);
//
// Convention.  A Tensor4D of shape (N, C, H, W) flattens to a Matrix
// of shape (C*H*W, N) -- one example per column, exactly matching the
// "examples-as-columns" rule used everywhere else in the library.
// Within a column, elements are ordered c -> h -> w (channel slowest,
// width fastest), matching the NCHW storage order of Tensor4D.
//
#include "Matrix.h"
#include "Tensor4D.h"

#include <cstddef>
#include <stdexcept>

namespace weft {

// Tensor4D(N, C, H, W) -> Matrix(C*H*W, N).
// Each example becomes a column; within a column, traversal is c -> h -> w.
template <typename T>
Matrix<T> flatten(const Tensor4D<T>& X) {
    const std::size_t N = X.N();
    const std::size_t F = X.C() * X.H() * X.W();
    Matrix<T> out(F, N);
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t c = 0; c < X.C(); ++c)
            for (std::size_t h = 0; h < X.H(); ++h)
                for (std::size_t w = 0; w < X.W(); ++w) {
                    const std::size_t row = (c * X.H() + h) * X.W() + w;
                    out(row, n) = X(n, c, h, w);
                }
    return out;
}

// Matrix(C*H*W, N) -> Tensor4D(N, C, H, W).
// Inverse of flatten(); the caller supplies the target shape (the conv
// stack's output shape, which is known from the forward pass).
template <typename T>
Tensor4D<T> unflatten(const Matrix<T>& M,
                      std::size_t N, std::size_t C,
                      std::size_t H, std::size_t W) {
    if (M.rows() != C * H * W)
        throw std::invalid_argument("unflatten: row count != C*H*W");
    if (M.cols() != N)
        throw std::invalid_argument("unflatten: col count != N");

    Tensor4D<T> X(N, C, H, W);
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t h = 0; h < H; ++h)
                for (std::size_t w = 0; w < W; ++w) {
                    const std::size_t row = (c * H + h) * W + w;
                    X(n, c, h, w) = M(row, n);
                }
    return X;
}

} // namespace weft
