#pragma once
//
// SGD.h  --  plain stochastic gradient descent.
//
//      param  <-  param  -  lr * grad
//
// No state, no momentum. Useful as a baseline and as a foil to understand
// what Adam buys.  If you need momentum, use Adam with beta2 set very
// high (effectively disabling the adaptive part); we don't ship a
// separate "SGD with momentum" because Adam subsumes it.
//
#include "Optimizer.h"
#include "Matrix.h"

namespace weft {

template <typename T = float>
class SGD : public Optimizer<T> {
public:
    explicit SGD(T learning_rate) : lr_(learning_rate) {}

    void step(Matrix<T>& param, const Matrix<T>& grad) override {
        param -= lr_ * grad;
    }

    T learning_rate() const { return lr_; }
    void set_learning_rate(T lr) { lr_ = lr; }

private:
    T lr_;
};

} // namespace weft
