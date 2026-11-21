// test_layer_norm.cpp  --  verify LayerNorm's forward statistics, its
// three gradients (input, gamma, beta) against finite differences in
// double precision, and the save/load round-trip.

#include "LayerNorm.h"
#include "Matrix.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace weft;

namespace {

template <typename T>
Matrix<T> rand_mat(std::size_t r, std::size_t c, unsigned seed,
                   T lo = T(-1), T hi = T(1))
{
    Matrix<T> M(r, c);
    M.randomizeUniform(lo, hi, seed);
    return M;
}

// Scalar loss  L = sum(Wt .* Y).  A plain sum(Y) would be useless here:
// with gamma = 1 the normalised columns always sum to zero, so the loss
// would not depend on X at all and any dX would "pass".  Random weights
// make every output entry matter differently.
template <typename T>
T weighted_sum(const Matrix<T>& Wt, const Matrix<T>& Y) {
    T s = T(0);
    for (std::size_t j = 0; j < Y.cols(); ++j)
        for (std::size_t i = 0; i < Y.rows(); ++i) s += Wt(i, j) * Y(i, j);
    return s;
}

int n_passed = 0, n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

} // namespace

// Fresh layer (gamma = 1, beta = 0): every output column has mean 0 and
// variance 1, whatever the scale and offset of the input column.
void test_forward_statistics() {
    std::cout << "forward statistics:\n";
    const std::size_t D = 16, N = 5;
    auto X = rand_mat<double>(D, N, 1, -3.0, 7.0);
    for (std::size_t i = 0; i < D; ++i) X(i, 2) = 100.0 * X(i, 2) + 40.0;

    LayerNorm<double> ln(D);
    auto Y = ln.forward(X);
    check(Y.rows() == D && Y.cols() == N, "output shape matches input");

    double worst_mean = 0.0, worst_var = 0.0;
    for (std::size_t j = 0; j < N; ++j) {
        double mu = 0.0, var = 0.0;
        for (std::size_t i = 0; i < D; ++i) mu += Y(i, j);
        mu /= D;
        for (std::size_t i = 0; i < D; ++i) var += (Y(i, j) - mu) * (Y(i, j) - mu);
        var /= D;
        worst_mean = std::max(worst_mean, std::fabs(mu));
        worst_var  = std::max(worst_var,  std::fabs(var - 1.0));
    }
    check(worst_mean < 1e-12, "every column has mean 0");
    check(worst_var  < 1e-4,  "every column has variance 1 (up to eps)");
}

// Columns are normalised independently: changing one column must not
// change the output of any other.
void test_columns_independent() {
    std::cout << "columns are independent:\n";
    const std::size_t D = 8, N = 4;
    auto X = rand_mat<double>(D, N, 2);
    LayerNorm<double> ln(D);
    auto Y0 = ln.forward(X);
    auto X2 = X;
    for (std::size_t i = 0; i < D; ++i) X2(i, 1) = 5.0 * X2(i, 1) - 2.0 * i;
    auto Y1 = ln.forward(X2);
    double other = 0.0, same = 0.0;
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < D; ++i) {
            const double e = std::fabs(Y1(i, j) - Y0(i, j));
            if (j == 1) same = std::max(same, e); else other = std::max(other, e);
        }
    check(other == 0.0, "other columns are bit-identical");
    check(same > 1e-3,  "the changed column does change (sanity)");
}

// gamma and beta act per feature (per row), after the normalisation.
void test_affine() {
    std::cout << "gamma / beta:\n";
    const std::size_t D = 6, N = 3;
    auto X = rand_mat<double>(D, N, 3);
    LayerNorm<double> plain(D), scaled(D);
    for (std::size_t i = 0; i < D; ++i) {
        scaled.gamma()(i, 0) = 0.5 + i;
        scaled.beta()(i, 0)  = -1.0 + 0.25 * i;
    }
    auto Y0 = plain.forward(X);
    auto Y1 = scaled.forward(X);
    double worst = 0.0;
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < D; ++i)
            worst = std::max(worst, std::fabs(
                Y1(i, j) - ((0.5 + i) * Y0(i, j) + (-1.0 + 0.25 * i))));
    check(worst < 1e-12, "Y = gamma * x_hat + beta, row by row");
}

// All three gradients against central finite differences.
void test_numerical_grads() {
    std::cout << "numerical gradients (double precision):\n";
    using Dbl = double;
    const std::size_t D = 7, N = 4;
    auto X  = rand_mat<Dbl>(D, N, 10);
    auto Wt = rand_mat<Dbl>(D, N, 11);

    LayerNorm<Dbl> ln(D);
    // Non-trivial gamma/beta so the affine part is exercised too.
    for (std::size_t i = 0; i < D; ++i) {
        ln.gamma()(i, 0) = 0.6 + 0.2 * i;
        ln.beta()(i, 0)  = 0.3 - 0.1 * i;
    }

    ln.forward(X);
    auto dX = ln.backward(Wt);
    const Matrix<Dbl> dgamma = ln.dgamma();
    const Matrix<Dbl> dbeta  = ln.dbeta();

    const Dbl eps = 1e-5;
    Dbl worst_x = 0.0, worst_g = 0.0, worst_b = 0.0;

    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < D; ++i) {
            auto Xp = X, Xm = X;
            Xp(i, j) += eps; Xm(i, j) -= eps;
            const Dbl num = (weighted_sum(Wt, ln.forward(Xp))
                           - weighted_sum(Wt, ln.forward(Xm))) / (2.0 * eps);
            worst_x = std::max(worst_x, std::fabs(num - dX(i, j)));
        }
    for (std::size_t i = 0; i < D; ++i) {
        const Dbl g0 = ln.gamma()(i, 0);
        ln.gamma()(i, 0) = g0 + eps; const Dbl lp = weighted_sum(Wt, ln.forward(X));
        ln.gamma()(i, 0) = g0 - eps; const Dbl lm = weighted_sum(Wt, ln.forward(X));
        ln.gamma()(i, 0) = g0;
        worst_g = std::max(worst_g,
                           std::fabs((lp - lm) / (2.0 * eps) - dgamma(i, 0)));

        const Dbl b0 = ln.beta()(i, 0);
        ln.beta()(i, 0) = b0 + eps; const Dbl bp = weighted_sum(Wt, ln.forward(X));
        ln.beta()(i, 0) = b0 - eps; const Dbl bm = weighted_sum(Wt, ln.forward(X));
        ln.beta()(i, 0) = b0;
        worst_b = std::max(worst_b,
                           std::fabs((bp - bm) / (2.0 * eps) - dbeta(i, 0)));
    }
    std::cout << "  worst |num - analytic|: dX " << worst_x
              << "  dgamma " << worst_g << "  dbeta " << worst_b << "\n";
    check(worst_x < 1e-8, "dX matches numerical");
    check(worst_g < 1e-8, "dgamma matches numerical");
    check(worst_b < 1e-8, "dbeta matches numerical");
}

void test_save_load_and_errors() {
    std::cout << "save/load and argument checks:\n";
    const std::size_t D = 5;
    LayerNorm<float> a(D), b(D);
    for (std::size_t i = 0; i < D; ++i) {
        a.gamma()(i, 0) = 1.5f + i;
        a.beta()(i, 0)  = -0.5f * i;
    }
    std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
    a.save_params(buf);
    b.load_params(buf);
    bool same = true;
    for (std::size_t i = 0; i < D; ++i)
        if (a.gamma()(i, 0) != b.gamma()(i, 0) || a.beta()(i, 0) != b.beta()(i, 0))
            same = false;
    check(same, "gamma and beta survive save -> load");

    bool caught = false;
    try {
        std::stringstream buf2(std::ios::in | std::ios::out | std::ios::binary);
        a.save_params(buf2);
        LayerNorm<float> wrong(D + 1);
        wrong.load_params(buf2);
    } catch (const std::runtime_error&) { caught = true; }
    check(caught, "load rejects a different d_model");

    caught = false;
    try { a.forward(Matrix<float>(D + 2, 3)); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "forward rejects input rows != d_model");

    check(a.describe() == "LayerNorm(d_model=5)", "describe string matches");
}

int main() {
    std::cout << "test_layer_norm\n";
    test_forward_statistics();
    test_columns_independent();
    test_affine();
    test_numerical_grads();
    test_save_load_and_errors();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
