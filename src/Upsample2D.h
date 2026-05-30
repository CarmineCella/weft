#pragma once
//
// Upsample2D.h  --  nearest-neighbour spatial upsampling.
//
// FORWARD.  Each input pixel (n, c, h, w) gets replicated into a
// factor x factor block in the output at the same channel:
//   Y(n, c, factor*h + dy, factor*w + dx)  =  X(n, c, h, w)
// for all dy, dx in [0, factor).  This is the simplest upsampler --
// no interpolation, just "make each pixel bigger."  The output has
// shape (N, C, H * factor, W * factor).
//
// BACKWARD.  Every output position (h_out, w_out) was a copy of
// input (h_out / factor, w_out / factor).  The gradient w.r.t. that
// input is the SUM of gradients at every output position that copied
// from it -- i.e., we sum over the factor x factor block.  No max,
// no selection: every output cell contributes.
//
// This is the natural decoder counterpart of MaxPool2D for VAE-style
// architectures: "encoder downsamples by 2 via MaxPool, decoder
// upsamples by 2 via Upsample".  We prefer Upsample + Conv2D over
// TransposedConv2D because transposed conv produces checkerboard
// artefacts when its stride doesn't divide its kernel size, which
// makes it a poor default for spectrogram generation.  Upsample +
// Conv2D avoids those artefacts entirely.
//
// No learnable parameters, so update() is a no-op.
//
#include "ConvLayer.h"
#include "Tensor4D.h"
#include "Optimizer.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace weft {

template <typename T = float>
class Upsample2D : public ConvLayer<T> {
public:
    // factor is the linear upscaling ratio; e.g. factor=2 doubles both
    // spatial dims.  factor=1 is identity but allowed for completeness.
    explicit Upsample2D(std::size_t factor)
        : factor_(factor) {
        if (factor == 0)
            throw std::invalid_argument("Upsample2D: factor must be > 0");
    }

    Tensor4D<T> forward(const Tensor4D<T>& X) override {
        cN_ = X.N(); cC_ = X.C(); cH_ = X.H(); cW_ = X.W();
        const std::size_t H_out = cH_ * factor_;
        const std::size_t W_out = cW_ * factor_;

        Tensor4D<T> Y(cN_, cC_, H_out, W_out);
        for (std::size_t n = 0; n < cN_; ++n)
            for (std::size_t c = 0; c < cC_; ++c)
                for (std::size_t h = 0; h < cH_; ++h)
                    for (std::size_t w = 0; w < cW_; ++w) {
                        const T v = X(n, c, h, w);
                        for (std::size_t dy = 0; dy < factor_; ++dy)
                            for (std::size_t dx = 0; dx < factor_; ++dx)
                                Y(n, c, h * factor_ + dy, w * factor_ + dx) = v;
                    }
        return Y;
    }

    Tensor4D<T> backward(const Tensor4D<T>& dY) override {
        Tensor4D<T> dX(cN_, cC_, cH_, cW_);     // zero-initialised
        for (std::size_t n = 0; n < cN_; ++n)
            for (std::size_t c = 0; c < cC_; ++c)
                for (std::size_t h = 0; h < cH_; ++h)
                    for (std::size_t w = 0; w < cW_; ++w) {
                        T acc = T(0);
                        for (std::size_t dy = 0; dy < factor_; ++dy)
                            for (std::size_t dx = 0; dx < factor_; ++dx)
                                acc += dY(n, c, h * factor_ + dy,
                                                w * factor_ + dx);
                        dX(n, c, h, w) = acc;
                    }
        return dX;
    }

    void update(Optimizer<T>&) override {}

    std::string describe() const override {
        std::ostringstream ss;
        ss << "Upsample2D(factor=" << factor_ << ")";
        return ss.str();
    }

    std::size_t factor() const { return factor_; }

private:
    std::size_t factor_;
    std::size_t cN_ = 0, cC_ = 0, cH_ = 0, cW_ = 0;
};

} // namespace weft
