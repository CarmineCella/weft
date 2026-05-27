#pragma once
//
// Adam.h  --  Adaptive Moment Estimation (Kingma & Ba, 2014).
//
// Combines two ideas:
//   - momentum  : an EMA of the gradient (1st moment),
//   - RMSProp   : an EMA of the squared gradient (2nd moment),
// plus a bias correction for the fact that both moments start at zero.
//
// Update rule, computed per parameter independently:
//      m_t = beta1 * m_{t-1} + (1 - beta1) * g
//      v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
//      m_hat = m_t / (1 - beta1^t)
//      v_hat = v_t / (1 - beta2^t)
//      param -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Typical hyperparameters:
//      lr     = 1e-3   (sometimes 3e-4 for very deep nets)
//      beta1  = 0.9    (EMA decay for the 1st moment)
//      beta2  = 0.999  (EMA decay for the 2nd moment)
//      eps    = 1e-8   (numerical guard for the sqrt)
//
// State (m, v, t) is allocated lazily the first time a given parameter is
// seen, and is keyed on the parameter's address.  As long as the layer's
// parameter Matrix objects don't move in memory (they don't, they're
// member variables of the layer), the same per-parameter state is found
// on every subsequent step.
//
#include "Optimizer.h"
#include "Matrix.h"

#include <cmath>
#include <cstddef>
#include <unordered_map>

namespace weft {

template <typename T = float>
class Adam : public Optimizer<T> {
public:
    explicit Adam(T learning_rate = T(1e-3),
                  T beta1         = T(0.9),
                  T beta2         = T(0.999),
                  T eps           = T(1e-8))
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {}

    void step(Matrix<T>& param, const Matrix<T>& grad) override {
        State& s = state_[&param];
        if (s.t == 0) {
            // Lazy init on first contact: allocate moments matching shape.
            s.m = Matrix<T>(param.rows(), param.cols(), T(0));
            s.v = Matrix<T>(param.rows(), param.cols(), T(0));
        }
        s.t += 1;

        // Bias-correction factors at step t.
        const T bc1 = T(1) - std::pow(beta1_, static_cast<T>(s.t));
        const T bc2 = T(1) - std::pow(beta2_, static_cast<T>(s.t));

        for (std::size_t i = 0; i < param.rows(); ++i) {
            for (std::size_t j = 0; j < param.cols(); ++j) {
                const T g = grad(i, j);
                s.m(i, j) = beta1_ * s.m(i, j) + (T(1) - beta1_) * g;
                s.v(i, j) = beta2_ * s.v(i, j) + (T(1) - beta2_) * g * g;
                const T m_hat = s.m(i, j) / bc1;
                const T v_hat = s.v(i, j) / bc2;
                param(i, j) -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }

    // For tests / introspection
    T learning_rate() const { return lr_; }
    void set_learning_rate(T lr) { lr_ = lr; }

private:
    struct State {
        Matrix<T> m;
        Matrix<T> v;
        int       t = 0;
    };
    std::unordered_map<Matrix<T>*, State> state_;
    T lr_, beta1_, beta2_, eps_;
};

} // namespace weft
