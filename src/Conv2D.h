#pragma once
//
// Conv2D.h  --  2-D convolutional layer.
//
// FORWARD via im2col + matrix multiply.
// -------------------------------------
// A 2-D convolution slides a small kernel (say 3x3) across the H-W
// plane of each input channel, multiplying element-wise and summing,
// producing one value per (output_channel, output_position).  The
// naive nested-loop implementation is six loops deep and slow.
//
// im2col rearranges the input so the whole convolution becomes a
// single matrix multiply:
//
//   * For every output position (n, h_out, w_out), the input patch
//     under the kernel at that position -- shape (in_C, K, K) -- is
//     flattened into one column of an "im2col" matrix.
//   * The kernel weights are reshaped from (out_C, in_C, K, K) into
//     a matrix W of shape (out_C, in_C*K*K).  Each row of W is one
//     output channel's flattened kernel.
//   * Multiplying  W (out_C x in_C*K*K)  by  col (in_C*K*K x N*H_out*W_out)
//     gives an output matrix of shape (out_C, N*H_out*W_out).  Reshape
//     it back to (N, out_C, H_out, W_out) and we have the conv output.
//
// The win is that Matrix::operator* is already parallelised (and BLAS-
// accelerated when WEFT_USE_BLAS is defined), so we inherit all that
// work for free.  This is the same trick cuDNN uses on GPU.  The cost
// is the temporary im2col matrix, which is roughly K*K times bigger
// than the input.  Fine at our scales.
//
// BACKWARD via col2im accumulation.
// ---------------------------------
// Given the upstream gradient dY of shape (N, out_C, H_out, W_out):
//
//   1.  Reshape dY into dZ of shape (out_C, N*H_out*W_out)  -- same
//       column-ordering convention as forward.
//   2.  dW = dZ * col^T                  (the kernel-gradient matrix)
//   3.  db = sum-along-columns of dZ     (one scalar per out channel)
//   4.  dCol = W^T * dZ                  (gradient w.r.t. im2col)
//   5.  col2im(dCol) -> dX:  fold dCol back into the (N, in_C, H, W)
//       input shape, ACCUMULATING at every (n, c, h_in, w_in) the
//       contributions from every output position whose receptive
//       field touched that input pixel.  This accumulation is the
//       part that's easy to get wrong; for stride < kernel_size, one
//       input pixel sits under many output positions.
//
// The numerical-gradient check in tests/test_conv2d.cpp is what
// guards against any indexing mistake in either im2col or col2im.
//
#include "ConvLayer.h"
#include "Matrix.h"
#include "Tensor4D.h"
#include "Optimizer.h"

#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <cstdint>

namespace weft {

template <typename T = float>
class Conv2D : public ConvLayer<T> {
public:
    Conv2D(std::size_t in_channels,
           std::size_t out_channels,
           std::size_t kernel_size,
           std::size_t stride  = 1,
           std::size_t padding = 0,
           unsigned    seed    = std::random_device{}())
        : in_C_(in_channels), out_C_(out_channels),
          K_(kernel_size), stride_(stride), padding_(padding),
          // Kernel as a (out_C, in_C*K*K) matrix: each row is one
          // output channel's flattened kernel.  Shaped this way, the
          // forward is just W * im2col.
          W_(out_channels, in_channels * kernel_size * kernel_size),
          b_(out_channels, 1, T(0)),
          dW_(out_channels, in_channels * kernel_size * kernel_size),
          db_(out_channels, 1) {
        if (kernel_size == 0) throw std::invalid_argument("Conv2D: kernel_size must be > 0");
        if (stride == 0)      throw std::invalid_argument("Conv2D: stride must be > 0");

        // He init: stddev = sqrt(2 / fan_in) for ReLU-fed layers.
        // fan_in is the number of inputs to one output activation:
        // in_C * K * K (one full receptive field).
        const T fan_in = static_cast<T>(in_channels * kernel_size * kernel_size);
        const T stddev = std::sqrt(T(2) / fan_in);
        W_.randomizeNormal(T(0), stddev, seed);
        b_.fill(T(0));
    }

    // ---------------------------------------------------------------
    // Forward.  Returns Y of shape (N, out_C, H_out, W_out).  Caches
    // the input shape and the im2col matrix for backward.
    // ---------------------------------------------------------------
    Tensor4D<T> forward(const Tensor4D<T>& X) override {
        if (X.C() != in_C_)
            throw std::invalid_argument("Conv2D::forward: input channel count mismatch");

        cached_N_ = X.N();
        cached_H_ = X.H();
        cached_W_ = X.W();

        const std::size_t H_out = out_dim(X.H());
        const std::size_t W_out = out_dim(X.W());

        col_cache_ = im2col(X, H_out, W_out);

        // Z = W * col + b, with b (out_C x 1) broadcast across columns.
        Matrix<T> Z = W_ * col_cache_ + b_;

        // Reshape Z (out_C, N*H_out*W_out) back to Tensor4D
        // (N, out_C, H_out, W_out).  Column ordering matches im2col:
        //   col_idx = n * (H_out * W_out) + h_out * W_out + w_out
        Tensor4D<T> Y(X.N(), out_C_, H_out, W_out);
        for (std::size_t n = 0; n < X.N(); ++n)
            for (std::size_t c = 0; c < out_C_; ++c)
                for (std::size_t h = 0; h < H_out; ++h)
                    for (std::size_t w = 0; w < W_out; ++w) {
                        const std::size_t col = n * (H_out * W_out) + h * W_out + w;
                        Y(n, c, h, w) = Z(c, col);
                    }
        return Y;
    }

    // ---------------------------------------------------------------
    // Backward.  Stashes dW, db on the layer; returns dX.
    // ---------------------------------------------------------------
    Tensor4D<T> backward(const Tensor4D<T>& dY) override {
        if (dY.C() != out_C_)
            throw std::invalid_argument("Conv2D::backward: dY channel count mismatch");

        const std::size_t N     = dY.N();
        const std::size_t H_out = dY.H();
        const std::size_t W_out = dY.W();

        // Reshape dY (N, out_C, H_out, W_out) -> dZ (out_C, N*H_out*W_out)
        // using the same column ordering as im2col.
        Matrix<T> dZ(out_C_, N * H_out * W_out);
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t c = 0; c < out_C_; ++c)
                for (std::size_t h = 0; h < H_out; ++h)
                    for (std::size_t w = 0; w < W_out; ++w) {
                        const std::size_t col = n * (H_out * W_out) + h * W_out + w;
                        dZ(c, col) = dY(n, c, h, w);
                    }

        // Parameter gradients.
        dW_ = dZ * col_cache_.transpose();        // (out_C, in_C*K*K)
        db_ = dZ.sumColumns();                    // (out_C, 1)

        // Gradient w.r.t. the im2col matrix, then fold back into input.
        Matrix<T> dCol = W_.transpose() * dZ;     // (in_C*K*K, N*H_out*W_out)
        return col2im(dCol, N, cached_H_, cached_W_, H_out, W_out);
    }

    void update(Optimizer<T>& opt) override {
        opt.step(W_, dW_);
        opt.step(b_, db_);
    }

    std::string describe() const override {
        std::ostringstream ss;
        ss << "Conv2D(" << in_C_ << "->" << out_C_
           << ", k=" << K_ << "x" << K_
           << ", s=" << stride_
           << ", p=" << padding_ << ")";
        return ss.str();
    }

    // Serialization mirrors Dense: rows/cols header + native-order floats.
    void save_params(std::ostream& os) const override {
        write_matrix(os, W_);
        write_matrix(os, b_);
    }
    void load_params(std::istream& is) override {
        read_matrix(is, W_, "Conv2D::W");
        read_matrix(is, b_, "Conv2D::b");
    }

    // Accessors for inspection (filter visualization, gradient checking).
    const Matrix<T>& W()  const { return W_; }
    const Matrix<T>& b()  const { return b_; }
          Matrix<T>& W()        { return W_; }
          Matrix<T>& b()        { return b_; }
    const Matrix<T>& dW() const { return dW_; }
    const Matrix<T>& db() const { return db_; }

    std::size_t in_channels()  const { return in_C_; }
    std::size_t out_channels() const { return out_C_; }
    std::size_t kernel_size()  const { return K_; }
    std::size_t stride()       const { return stride_; }
    std::size_t padding()      const { return padding_; }

private:
    std::size_t in_C_, out_C_, K_, stride_, padding_;
    Matrix<T>   W_,  b_;
    Matrix<T>   dW_, db_;
    Matrix<T>   col_cache_;
    std::size_t cached_N_ = 0, cached_H_ = 0, cached_W_ = 0;

    std::size_t out_dim(std::size_t in_dim) const {
        const long n = static_cast<long>(in_dim) + 2*static_cast<long>(padding_)
                      - static_cast<long>(K_);
        if (n < 0) throw std::invalid_argument("Conv2D: input + 2*padding < kernel_size");
        return static_cast<std::size_t>(n) / stride_ + 1;
    }

    // ---------------------------------------------------------------
    // im2col: build a (in_C*K*K, N*H_out*W_out) matrix where each
    // column is one flattened receptive-field patch from the input.
    //
    // Column-ordering convention (must match the forward reshape and
    // the backward dY-reshape):
    //     col_idx = n * (H_out * W_out) + h_out * W_out + w_out
    // Row-ordering within a column:
    //     row     = c * (K * K) + kh * K + kw
    // ---------------------------------------------------------------
    Matrix<T> im2col(const Tensor4D<T>& X,
                     std::size_t H_out, std::size_t W_out) const {
        const std::size_t N = X.N();
        Matrix<T> col(in_C_ * K_ * K_, N * H_out * W_out, T(0));

        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t h_out = 0; h_out < H_out; ++h_out)
                for (std::size_t w_out = 0; w_out < W_out; ++w_out) {
                    const std::size_t col_idx =
                        n * (H_out * W_out) + h_out * W_out + w_out;
                    for (std::size_t c = 0; c < in_C_; ++c)
                        for (std::size_t kh = 0; kh < K_; ++kh)
                            for (std::size_t kw = 0; kw < K_; ++kw) {
                                const long h_in = static_cast<long>(h_out * stride_)
                                                + static_cast<long>(kh)
                                                - static_cast<long>(padding_);
                                const long w_in = static_cast<long>(w_out * stride_)
                                                + static_cast<long>(kw)
                                                - static_cast<long>(padding_);
                                const std::size_t row = c * (K_ * K_) + kh * K_ + kw;
                                if (h_in >= 0 && h_in < static_cast<long>(X.H()) &&
                                    w_in >= 0 && w_in < static_cast<long>(X.W())) {
                                    col(row, col_idx) =
                                        X(n, c, static_cast<std::size_t>(h_in),
                                                static_cast<std::size_t>(w_in));
                                }
                                // outside input bounds -> stays 0 (= zero-padding)
                            }
                }
        return col;
    }

    // ---------------------------------------------------------------
    // col2im: fold an (in_C*K*K, N*H_out*W_out) gradient matrix back
    // into a (N, in_C, H, W) tensor, ACCUMULATING contributions at
    // every input position covered by multiple receptive fields.
    //
    // Uses the same row/column orderings as im2col.  Out-of-bounds
    // rows (corresponding to padding) are simply skipped -- there is
    // no input pixel there to receive a gradient.
    // ---------------------------------------------------------------
    Tensor4D<T> col2im(const Matrix<T>& dCol,
                       std::size_t N, std::size_t H, std::size_t W,
                       std::size_t H_out, std::size_t W_out) const {
        Tensor4D<T> dX(N, in_C_, H, W);   // zero-initialised

        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t h_out = 0; h_out < H_out; ++h_out)
                for (std::size_t w_out = 0; w_out < W_out; ++w_out) {
                    const std::size_t col_idx =
                        n * (H_out * W_out) + h_out * W_out + w_out;
                    for (std::size_t c = 0; c < in_C_; ++c)
                        for (std::size_t kh = 0; kh < K_; ++kh)
                            for (std::size_t kw = 0; kw < K_; ++kw) {
                                const long h_in = static_cast<long>(h_out * stride_)
                                                + static_cast<long>(kh)
                                                - static_cast<long>(padding_);
                                const long w_in = static_cast<long>(w_out * stride_)
                                                + static_cast<long>(kw)
                                                - static_cast<long>(padding_);
                                const std::size_t row = c * (K_ * K_) + kh * K_ + kw;
                                if (h_in >= 0 && h_in < static_cast<long>(H) &&
                                    w_in >= 0 && w_in < static_cast<long>(W)) {
                                    dX(n, c,
                                       static_cast<std::size_t>(h_in),
                                       static_cast<std::size_t>(w_in))
                                        += dCol(row, col_idx);
                                }
                            }
                }
        return dX;
    }

    // Same on-disk matrix format as Dense uses.
    static void write_matrix(std::ostream& os, const Matrix<T>& M) {
        std::uint32_t r = static_cast<std::uint32_t>(M.rows());
        std::uint32_t c = static_cast<std::uint32_t>(M.cols());
        os.write(reinterpret_cast<const char*>(&r), sizeof(r));
        os.write(reinterpret_cast<const char*>(&c), sizeof(c));
        os.write(reinterpret_cast<const char*>(M.data()), sizeof(T) * r * c);
    }
    static void read_matrix(std::istream& is, Matrix<T>& M, const char* name) {
        std::uint32_t r, c;
        is.read(reinterpret_cast<char*>(&r), sizeof(r));
        is.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (r != M.rows() || c != M.cols())
            throw std::runtime_error(std::string(name) + ": shape mismatch on load");
        is.read(reinterpret_cast<char*>(M.data()), sizeof(T) * r * c);
    }
};

} // namespace weft
