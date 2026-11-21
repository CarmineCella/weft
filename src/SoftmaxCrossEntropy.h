#pragma once
//
// Categorical cross-entropy directly from logits (classes x batch).
// Log-sum-exp avoids the probability-space epsilon floor. backward()
// returns dL/d(logits), including the mean-over-batch factor.
//
#include "Loss.h"
#include "Matrix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <utility>

namespace weft {

template <typename T = float>
class SoftmaxCrossEntropy : public Loss<T> {
public:
    T forward(const Matrix<T>& logits, const Matrix<T>& target) override {
        ready_ = false;
        if (logits.rows() == 0 || logits.cols() == 0 ||
            logits.rows() != target.rows() || logits.cols() != target.cols())
            throw std::invalid_argument("SoftmaxCrossEntropy: nonempty, matching shapes required");

        Matrix<T> probabilities(logits.rows(), logits.cols());
        std::vector<T> mass(logits.cols(), T(0));
        long double total = 0;
        for (std::size_t n = 0; n < logits.cols(); ++n) {
            T maximum = logits(0, n);
            for (std::size_t i = 0; i < logits.rows(); ++i) {
                if (!std::isfinite(logits(i, n)) ||
                    !std::isfinite(target(i, n)) || target(i, n) < T(0))
                    throw std::invalid_argument("SoftmaxCrossEntropy: invalid logits or targets");
                maximum = std::max(maximum, logits(i, n));
                mass[n] += target(i, n);
            }
            if (std::abs(mass[n] - T(1)) > T(1e-5))
                throw std::invalid_argument("SoftmaxCrossEntropy: target columns must sum to one");
            T denominator = T(0);
            for (std::size_t i = 0; i < logits.rows(); ++i) {
                probabilities(i, n) = std::exp(logits(i, n) - maximum);
                denominator += probabilities(i, n);
            }
            const T log_sum = std::log(denominator);
            for (std::size_t i = 0; i < logits.rows(); ++i) {
                probabilities(i, n) /= denominator;
                // Subtract the maximum before the logarithm is added, so
                // a large common shift does not cancel the log-sum term.
                total += static_cast<long double>(target(i, n)) *
                         ((static_cast<long double>(maximum) - logits(i, n)) + log_sum);
            }
        }
        probabilities_ = std::move(probabilities);
        target_ = target;
        target_mass_ = std::move(mass);
        ready_ = true;
        return static_cast<T>(total / logits.cols());
    }

    Matrix<T> backward() override {
        require_forward();
        Matrix<T> gradient(probabilities_.rows(), probabilities_.cols());
        const T scale = T(1) / static_cast<T>(gradient.cols());
        for (std::size_t n = 0; n < gradient.cols(); ++n)
            for (std::size_t i = 0; i < gradient.rows(); ++i)
                gradient(i, n) = (probabilities_(i, n) * target_mass_[n] -
                                  target_(i, n)) * scale;
        return gradient;
    }

    const Matrix<T>& probabilities() const {
        require_forward();
        return probabilities_;
    }

private:
    void require_forward() const {
        if (!ready_) throw std::logic_error("SoftmaxCrossEntropy: call forward first");
    }
    bool ready_ = false;
    Matrix<T> probabilities_, target_;
    std::vector<T> target_mass_;
};

} // namespace weft
