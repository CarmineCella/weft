#pragma once
//
// Standardizer.h  --  per-feature mean/std normalisation.
//
//   fit(X)              : learn per-feature mean and std from training data.
//   transform(X)        : return (X - mean) / std.
//   fit_transform(X)    : convenience -- fit then transform on the same X.
//   inverse_transform(Y): return Y * std + mean  (used to denormalise, e.g.
//                                                 autoencoder reconstructions).
//
// IMPORTANT: always fit on the TRAINING set only, then transform both train
// and test with the same Standardizer.  Refitting on test data leaks
// information about the test distribution into preprocessing and gives
// optimistic-but-wrong evaluation numbers.  This is the most common subtle
// bug in ML pipelines; the API is shaped to keep you on the right path.
//
// Shapes: X is (features x examples).  mean and std are (features x 1).
//
#include "Matrix.h"

#include <cmath>
#include <cstddef>

namespace weft {

template <typename T = float>
class Standardizer {
public:
    void fit(const Matrix<T>& X) {
        const std::size_t F = X.rows();
        const std::size_t N = X.cols();
        mean_ = Matrix<T>(F, 1);
        std_  = Matrix<T>(F, 1);

        // mean per feature (per row)
        for (std::size_t i = 0; i < F; ++i) {
            T sum = T(0);
            for (std::size_t j = 0; j < N; ++j) sum += X(i, j);
            mean_(i, 0) = sum / static_cast<T>(N);
        }
        // std per feature (population std; the divisor is N, not N-1)
        for (std::size_t i = 0; i < F; ++i) {
            T sumsq = T(0);
            for (std::size_t j = 0; j < N; ++j) {
                T d = X(i, j) - mean_(i, 0);
                sumsq += d * d;
            }
            T s = std::sqrt(sumsq / static_cast<T>(N));
            // Guard against constant features (would otherwise produce 0/0)
            std_(i, 0) = (s < T(1e-7)) ? T(1) : s;
        }
    }

    Matrix<T> transform(const Matrix<T>& X) const {
        if (X.rows() != mean_.rows())
            throw std::invalid_argument(
                "Standardizer::transform: feature count does not match fit()");
        Matrix<T> Y(X.rows(), X.cols());
        for (std::size_t i = 0; i < X.rows(); ++i) {
            const T m = mean_(i, 0);
            const T s = std_(i, 0);
            for (std::size_t j = 0; j < X.cols(); ++j)
                Y(i, j) = (X(i, j) - m) / s;
        }
        return Y;
    }

    Matrix<T> fit_transform(const Matrix<T>& X) {
        fit(X);
        return transform(X);
    }

    // Undo the transformation.  Handy for autoencoder reconstruction or
    // anywhere we need the model output back on the original scale.
    Matrix<T> inverse_transform(const Matrix<T>& Y) const {
        if (Y.rows() != mean_.rows())
            throw std::invalid_argument(
                "Standardizer::inverse_transform: feature count does not match fit()");
        Matrix<T> X(Y.rows(), Y.cols());
        for (std::size_t i = 0; i < Y.rows(); ++i) {
            const T m = mean_(i, 0);
            const T s = std_(i, 0);
            for (std::size_t j = 0; j < Y.cols(); ++j)
                X(i, j) = Y(i, j) * s + m;
        }
        return X;
    }

    const Matrix<T>& mean() const { return mean_; }
    const Matrix<T>& std()  const { return std_;  }

private:
    Matrix<T> mean_;
    Matrix<T> std_;
};

} // namespace weft
