#pragma once
//
// ReLU4D.h  --  rectified-linear activation on Tensor4D.
//
// Same arithmetic as the matrix ReLU -- max(0, x) elementwise -- but on
// 4-D tensor data.  Activation layers are storage-shape-agnostic, so
// the implementation just walks the flat data via apply(), exactly like
// ReLU does on Matrix.
//
// Backward routes the upstream gradient only through the entries that
// were positive in the forward pass:  dx = dy * (x > 0).
//
#include "ConvLayer.h"
#include "Tensor4D.h"
#include "Optimizer.h"

namespace weft {

template <typename T = float>
class ReLU4D : public ConvLayer<T> {
public:
    Tensor4D<T> forward(const Tensor4D<T>& X) override {
        cache_ = X;
        return X.apply([](T v) { return v > T(0) ? v : T(0); });
    }

    Tensor4D<T> backward(const Tensor4D<T>& dY) override {
        // Mask: 1 where cached input was positive, 0 elsewhere.
        Tensor4D<T> mask = cache_.apply([](T v) { return v > T(0) ? T(1) : T(0); });
        return hadamard(dY, mask);
    }

    void update(Optimizer<T>&) override {}        // no parameters
    std::string describe() const override { return "ReLU4D"; }

private:
    Tensor4D<T> cache_;
};

} // namespace weft
