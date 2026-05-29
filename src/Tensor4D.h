#pragma once
//
// Tensor4D.h  --  4-D dense tensor for convolutional layers.
//
// CONVENTIONS
// -----------
// Shape (NCHW):     (batch, channels, height, width)
//                    same as PyTorch and cuDNN.
//
// Storage:           single flat std::vector<T> in row-major order.
//                    Element (n, c, h, w) lives at
//                        data_[((n*C + c)*H + h)*W + w]
//                    so walking the flat array linearly walks W fastest,
//                    then H, then C, then N -- meaning consecutive
//                    pixels of one channel of one image are adjacent.
//                    That is what 2-D spatial convolution wants.
//
// Tensor4D is a SEPARATE type from Matrix.  Conv layers operate on
// Tensor4D; dense layers operate on Matrix.  At the conv-to-dense
// boundary (the classic AlexNet shape:   conv ... conv -> flatten ->
// dense ... dense -> softmax) a free function `flatten` turns a
// Tensor4D of shape (N, C, H, W) into a Matrix of shape (C*H*W, N) --
// features down, examples across, matching Matrix's existing
// one-example-per-column convention.
//
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace weft {

template <typename T = float>
class Tensor4D {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------
    Tensor4D() : N_(0), C_(0), H_(0), W_(0) {}

    Tensor4D(std::size_t N, std::size_t C,
             std::size_t H, std::size_t W,
             T value = T(0))
        : N_(N), C_(C), H_(H), W_(W),
          data_(N * C * H * W, value) {}

    // ------------------------------------------------------------------
    // Shape
    // ------------------------------------------------------------------
    std::size_t N() const { return N_; }
    std::size_t C() const { return C_; }
    std::size_t H() const { return H_; }
    std::size_t W() const { return W_; }
    std::size_t size() const { return data_.size(); }

    // ------------------------------------------------------------------
    // Element access.  Two versions so it works on both mutable and
    // const tensors.  Equivalent of Matrix::operator()(i, j), but with
    // four indices.
    //     T(n, c, h, w) = ...;     x = T(n, c, h, w);
    // ------------------------------------------------------------------
    T& operator()(std::size_t n, std::size_t c,
                  std::size_t h, std::size_t w) {
        return data_[((n * C_ + c) * H_ + h) * W_ + w];
    }
    const T& operator()(std::size_t n, std::size_t c,
                        std::size_t h, std::size_t w) const {
        return data_[((n * C_ + c) * H_ + h) * W_ + w];
    }

    // Raw data access for inner loops (im2col, conv backward, etc.).
    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    // ------------------------------------------------------------------
    // Fill / random
    // ------------------------------------------------------------------
    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

    // Gaussian -- the usual starting point for conv kernels (He init
    // will pick the stddev when we get to Conv2D).
    void randomizeNormal(T mean = T(0), T stddev = T(1),
                         unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        std::normal_distribution<T> dist(mean, stddev);
        for (T& x : data_) x = dist(gen);
    }

    void randomizeUniform(T lo = T(-1), T hi = T(1),
                          unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> dist(lo, hi);
        for (T& x : data_) x = dist(gen);
    }

    // ------------------------------------------------------------------
    // Element-wise function application.  This is how 4-D activations
    // (ReLU4D, etc.) are implemented -- the activation doesn't care
    // about the (n, c, h, w) layout, it just walks the flat storage.
    // ------------------------------------------------------------------
    template <typename Func>
    Tensor4D apply(Func f) const {
        Tensor4D R(N_, C_, H_, W_);
        for (std::size_t k = 0; k < data_.size(); ++k)
            R.data_[k] = f(data_[k]);
        return R;
    }

    template <typename Func>
    void applyInPlace(Func f) {
        for (T& x : data_) x = f(x);
    }

    // ------------------------------------------------------------------
    // Debug printing: dump one channel of one batch element at a time.
    // Compact enough to be useful, structured enough to read.
    // ------------------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const Tensor4D& X) {
        os << "Tensor4D(" << X.N_ << " x " << X.C_ << " x "
           << X.H_ << " x " << X.W_ << ")\n";
        for (std::size_t n = 0; n < X.N_; ++n)
            for (std::size_t c = 0; c < X.C_; ++c) {
                os << "  [n=" << n << ", c=" << c << "]\n";
                for (std::size_t h = 0; h < X.H_; ++h) {
                    os << "   ";
                    for (std::size_t w = 0; w < X.W_; ++w)
                        os << std::setw(8) << std::setprecision(4)
                           << X(n, c, h, w) << ' ';
                    os << '\n';
                }
            }
        return os;
    }

private:
    std::size_t N_, C_, H_, W_;
    std::vector<T> data_;
};

// ----------------------------------------------------------------------
// Free functions
// ----------------------------------------------------------------------

// Element-wise (Hadamard) product of two same-shape tensors.  Used in
// element-wise backward computations (ReLU4D's backward, etc.), just
// like the existing hadamard() for Matrix.
template <typename T>
Tensor4D<T> hadamard(const Tensor4D<T>& A, const Tensor4D<T>& B) {
    if (A.N() != B.N() || A.C() != B.C() ||
        A.H() != B.H() || A.W() != B.W())
        throw std::invalid_argument("hadamard(Tensor4D): shape mismatch");
    Tensor4D<T> R(A.N(), A.C(), A.H(), A.W());
    const T* a = A.data();
    const T* b = B.data();
    T*       r = R.data();
    for (std::size_t k = 0; k < A.size(); ++k) r[k] = a[k] * b[k];
    return R;
}

} // namespace weft
