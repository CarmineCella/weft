#pragma once
//
// Dropout.h  --  randomly zero a fraction of activations during training.
//
// "Inverted dropout":  during training, each activation is independently
// either dropped (set to 0) with probability `rate` or kept and scaled by
// 1/(1 - rate).  The scaling makes the expected value of the output equal
// to the input, which means we can leave the inference path as plain
// identity.
//
// Mathematically, for each element x of the input,
//
//     y  =  m * x
//
// where m is a random variable taking the value 0 with probability `rate`
// and the value 1/(1 - rate) with probability (1 - rate).  Then
//
//     E[m] = rate * 0 + (1 - rate) * 1/(1 - rate) = 1   so   E[y] = x.
//
// Backward is dY * m  with the same mask used in forward -- structurally
// identical to ReLU's backward, just with a random mask instead of a
// data-driven one.
//
// During inference (training() == false), forward and backward are both
// the identity.  The Layer<T> training flag is flipped by Network::eval()
// and Network::train(), so the typical usage is:
//
//     net.train();
//     ... training loop ...
//     net.eval();
//     auto preds = net.forward(X_test);
//
#include "Layer.h"
#include "Matrix.h"

#include <random>

namespace weft {

template <typename T = float>
class Dropout : public Layer<T> {
public:
    // rate is the probability of dropping each activation; 0 disables
    // dropout entirely (forward is then identity in both modes).
    explicit Dropout(T rate, unsigned seed = std::random_device{}())
        : rate_(rate), gen_(seed) {}

    Matrix<T> forward(const Matrix<T>& X) override {
        if (!this->training() || rate_ == T(0))
            return X;

        const T scale = T(1) / (T(1) - rate_);
        std::bernoulli_distribution keep(static_cast<double>(T(1) - rate_));

        mask_ = Matrix<T>(X.rows(), X.cols());
        for (std::size_t i = 0; i < X.rows(); ++i)
            for (std::size_t j = 0; j < X.cols(); ++j)
                mask_(i, j) = keep(gen_) ? scale : T(0);

        return hadamard(X, mask_);
    }

    Matrix<T> backward(const Matrix<T>& dY) override {
        if (!this->training() || rate_ == T(0))
            return dY;
        return hadamard(dY, mask_);
    }

    T rate() const { return rate_; }

private:
    T rate_;
    std::mt19937 gen_;
    Matrix<T> mask_;
};

} // namespace weft
