#pragma once
//
// MaxPool2D.h  --  spatial max pooling.
//
// FORWARD.  For each output position (n, c, h_out, w_out), look at the
// pool_size x pool_size window of the input in that channel starting
// at (h_out*stride, w_out*stride), and emit the maximum value in that
// window.  Also remember WHICH input position was the maximum.
//
// BACKWARD.  The max operation is "select one entry, ignore the rest."
// So the gradient flows ONLY to the position that was the max during
// forward; every other position in the window contributed nothing to
// the output and receives no gradient.  We use the cached argmax to
// route dY back to exactly one input pixel per output pixel.
//
// No learnable parameters, so update() is a no-op.
//
// (Note: if two values in a pool window are exactly equal, the
// "winner" is whichever came first in the scan order.  In practice
// this never matters numerically; gradient checks should use random
// continuous inputs so ties have measure zero.)
//
#include "ConvLayer.h"
#include "Tensor4D.h"
#include "Optimizer.h"

#include <vector>
#include <cstddef>
#include <string>
#include <sstream>
#include <stdexcept>

namespace weft {

template <typename T = float>
class MaxPool2D : public ConvLayer<T> {
public:
    // stride defaults to pool_size: non-overlapping windows, which is the
    // typical "halve the spatial dimensions" use.
    MaxPool2D(std::size_t pool_size, std::size_t stride = 0)
        : pool_(pool_size),
          stride_(stride == 0 ? pool_size : stride) {
        if (pool_size == 0) throw std::invalid_argument("MaxPool2D: pool_size must be > 0");
    }

    Tensor4D<T> forward(const Tensor4D<T>& X) override {
        if (X.H() < pool_ || X.W() < pool_)
            throw std::invalid_argument("MaxPool2D::forward: input smaller than pool window");

        cN_ = X.N(); cC_ = X.C(); cH_ = X.H(); cW_ = X.W();
        const std::size_t H_out = (X.H() - pool_) / stride_ + 1;
        const std::size_t W_out = (X.W() - pool_) / stride_ + 1;

        Tensor4D<T> Y(cN_, cC_, H_out, W_out);
        // For each output cell, remember the LINEAR (h*W + w) index in
        // the input plane that won.  Flat indexing keeps this compact.
        max_idx_.assign(cN_ * cC_ * H_out * W_out, 0);

        for (std::size_t n = 0; n < cN_; ++n)
            for (std::size_t c = 0; c < cC_; ++c)
                for (std::size_t h_out = 0; h_out < H_out; ++h_out)
                    for (std::size_t w_out = 0; w_out < W_out; ++w_out) {
                        const std::size_t h0 = h_out * stride_;
                        const std::size_t w0 = w_out * stride_;
                        // Seed with the top-left of the window, then sweep.
                        T best = X(n, c, h0, w0);
                        std::size_t best_idx = h0 * cW_ + w0;
                        for (std::size_t kh = 0; kh < pool_; ++kh)
                            for (std::size_t kw = 0; kw < pool_; ++kw) {
                                const std::size_t h = h0 + kh;
                                const std::size_t w = w0 + kw;
                                const T v = X(n, c, h, w);
                                if (v > best) {
                                    best = v;
                                    best_idx = h * cW_ + w;
                                }
                            }
                        Y(n, c, h_out, w_out) = best;
                        const std::size_t flat =
                            ((n * cC_ + c) * H_out + h_out) * W_out + w_out;
                        max_idx_[flat] = best_idx;
                    }
        return Y;
    }

    Tensor4D<T> backward(const Tensor4D<T>& dY) override {
        const std::size_t H_out = dY.H();
        const std::size_t W_out = dY.W();
        Tensor4D<T> dX(cN_, cC_, cH_, cW_);   // zero-initialised

        for (std::size_t n = 0; n < cN_; ++n)
            for (std::size_t c = 0; c < cC_; ++c)
                for (std::size_t h_out = 0; h_out < H_out; ++h_out)
                    for (std::size_t w_out = 0; w_out < W_out; ++w_out) {
                        const std::size_t flat =
                            ((n * cC_ + c) * H_out + h_out) * W_out + w_out;
                        const std::size_t in_pos = max_idx_[flat];
                        const std::size_t h_in = in_pos / cW_;
                        const std::size_t w_in = in_pos % cW_;
                        // += rather than = because, for stride < pool_size,
                        // a single input pixel can win in multiple windows
                        // and should accumulate gradients from each.
                        dX(n, c, h_in, w_in) += dY(n, c, h_out, w_out);
                    }
        return dX;
    }

    void update(Optimizer<T>&) override {}

    std::string describe() const override {
        std::ostringstream ss;
        ss << "MaxPool2D(k=" << pool_ << ", s=" << stride_ << ")";
        return ss.str();
    }

    std::size_t pool_size() const { return pool_; }
    std::size_t stride()    const { return stride_; }

private:
    std::size_t pool_, stride_;
    std::size_t cN_ = 0, cC_ = 0, cH_ = 0, cW_ = 0;
    std::vector<std::size_t> max_idx_;
};

} // namespace weft
