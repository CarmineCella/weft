#pragma once
//
// LayerNorm.h  --  per-token, feature-axis normalisation with learnable
//                  affine scale (gamma) and shift (beta).
//
// Forward (per column j of X):
//
//   mu_j      = (1/D) sum_i  X[i, j]
//   var_j     = (1/D) sum_i  (X[i, j] - mu_j)^2
//   x_hat[i,j]= (X[i, j] - mu_j) / sqrt(var_j + eps)
//   Y[i, j]   = gamma[i] * x_hat[i, j] + beta[i]
//
// where D = X.rows() is the feature dimension and the mean/variance are
// computed independently for each example (column).  In contrast to
// BatchNorm, which averages across the *batch* dimension, LayerNorm's
// statistics depend only on the single column, which is why transformers
// like it: it works for any batch size including 1, and behaves the
// same at train- and inference-time without running averages.
//
// Backward.  The chain rule has three paths because X[k, j] influences
// x_hat[i, j] via the direct term, via mu_j, and via var_j.  Letting
//   dx_hat[i, j] = dY[i, j] * gamma[i]
//   S1_j         = sum_i dx_hat[i, j]
//   S2_j         = sum_i dx_hat[i, j] * x_hat[i, j]
//   f_j          = sqrt(var_j + eps)
// the standard derivation gives
//   dX[i, j]     = (1 / (D * f_j)) * (D * dx_hat[i, j] - S1_j
//                                     - x_hat[i, j] * S2_j)
//   dgamma[i]    = sum_j dY[i, j] * x_hat[i, j]    (summed across batch)
//   dbeta[i]     = sum_j dY[i, j]                  (summed across batch)
//
// Verified by finite differences in tests/test_layer_norm.cpp.
//
#include "Layer.h"
#include "Matrix.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

template <typename T = float>
class LayerNorm : public Layer<T> {
public:
    // d_model   feature dimension (must equal rows of every input)
    // eps       numerical stability constant; 1e-5 is the standard
    LayerNorm(std::size_t d_model, T eps = T(1e-5))
        : d_model_(d_model), eps_(eps),
          gamma_(d_model, 1, T(1)),     // start as identity affine
          beta_ (d_model, 1, T(0)),
          dgamma_(d_model, 1, T(0)),
          dbeta_ (d_model, 1, T(0))
    {}

    Matrix<T> forward(const Matrix<T>& X) override {
        if (X.rows() != d_model_)
            throw std::invalid_argument(
                "LayerNorm::forward: input rows must equal d_model");

        const std::size_t D = X.rows();
        const std::size_t N = X.cols();

        x_hat_   = Matrix<T>(D, N);
        inv_std_ = Matrix<T>(1, N);     // 1 / sqrt(var + eps) per column
        Matrix<T> Y(D, N);

        for (std::size_t j = 0; j < N; ++j) {
            // Mean and variance of column j.
            T mu = T(0);
            for (std::size_t i = 0; i < D; ++i) mu += X(i, j);
            mu /= static_cast<T>(D);

            T var = T(0);
            for (std::size_t i = 0; i < D; ++i) {
                const T d = X(i, j) - mu;
                var += d * d;
            }
            var /= static_cast<T>(D);

            const T inv = T(1) / std::sqrt(var + eps_);
            inv_std_(0, j) = inv;

            for (std::size_t i = 0; i < D; ++i) {
                x_hat_(i, j) = (X(i, j) - mu) * inv;
                Y(i, j)      = gamma_(i, 0) * x_hat_(i, j) + beta_(i, 0);
            }
        }
        return Y;
    }

    Matrix<T> backward(const Matrix<T>& dY) override {
        const std::size_t D = dY.rows();
        const std::size_t N = dY.cols();

        // Parameter gradients accumulate across the batch.
        for (std::size_t i = 0; i < D; ++i) {
            dgamma_(i, 0) = T(0);
            dbeta_ (i, 0) = T(0);
        }
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < D; ++i) {
                dgamma_(i, 0) += dY(i, j) * x_hat_(i, j);
                dbeta_ (i, 0) += dY(i, j);
            }

        // Input gradient: chain through (mu, var) per column.
        Matrix<T> dX(D, N);
        std::vector<T> dxhat(D);
        const T invD = T(1) / static_cast<T>(D);
        for (std::size_t j = 0; j < N; ++j) {
            const T inv = inv_std_(0, j);
            T S1 = T(0), S2 = T(0);
            for (std::size_t i = 0; i < D; ++i) {
                dxhat[i] = dY(i, j) * gamma_(i, 0);
                S1      += dxhat[i];
                S2      += dxhat[i] * x_hat_(i, j);
            }
            for (std::size_t i = 0; i < D; ++i)
                dX(i, j) = inv * (dxhat[i] - invD * S1 - invD * x_hat_(i, j) * S2);
        }
        return dX;
    }

    void update(Optimizer<T>& opt) override {
        opt.step(gamma_, dgamma_);
        opt.step(beta_,  dbeta_);
    }

    std::string describe() const override {
        return "LayerNorm(d_model=" + std::to_string(d_model_) + ")";
    }

    void save_params(std::ostream& out) const override {
        write_vec(out, gamma_);
        write_vec(out, beta_);
    }
    void load_params(std::istream& in) override {
        read_vec(in, gamma_);
        read_vec(in, beta_);
    }

    // Accessors for tests / inspection.
    const Matrix<T>& gamma()  const { return gamma_; }
    const Matrix<T>& beta()   const { return beta_;  }
    const Matrix<T>& dgamma() const { return dgamma_;}
    const Matrix<T>& dbeta()  const { return dbeta_; }
    std::size_t d_model()     const { return d_model_; }

private:
    static void write_vec(std::ostream& out, const Matrix<T>& V) {
        const std::uint32_t r = static_cast<std::uint32_t>(V.rows());
        out.write(reinterpret_cast<const char*>(&r), sizeof(r));
        for (std::size_t i = 0; i < V.rows(); ++i) {
            const T v = V(i, 0);
            out.write(reinterpret_cast<const char*>(&v), sizeof(T));
        }
    }
    static void read_vec(std::istream& in, Matrix<T>& V) {
        std::uint32_t r = 0;
        in.read(reinterpret_cast<char*>(&r), sizeof(r));
        if (r != V.rows())
            throw std::runtime_error(
                "LayerNorm::load_params: shape mismatch (build the same "
                "architecture before loading)");
        for (std::size_t i = 0; i < V.rows(); ++i) {
            T v = T(0);
            in.read(reinterpret_cast<char*>(&v), sizeof(T));
            V(i, 0) = v;
        }
    }

    std::size_t d_model_;
    T           eps_;
    Matrix<T>   gamma_, beta_;
    Matrix<T>   dgamma_, dbeta_;
    Matrix<T>   x_hat_;        // cached for backward
    Matrix<T>   inv_std_;      // 1/sqrt(var+eps) per column, cached
};

} // namespace weft
