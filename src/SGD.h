#pragma once
//
// SGD.h  --  plain stochastic gradient descent.
//
//      param  <-  param  -  lr * grad
//
// No state, no momentum. Useful as a baseline and as a foil to understand
// what Adam buys.  weft does not ship SGD with momentum.  Adam (Adam.h)
// contains a momentum term, but it also rescales every parameter by its
// own gradient history, so it is not a drop-in equivalent.
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
