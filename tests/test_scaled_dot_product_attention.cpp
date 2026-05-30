// test_scaled_dot_product_attention.cpp  --  verify forward + backward
// of ScaledDotProductAttention, including finite-difference gradient
// checks in double precision.

#include "Matrix.h"
#include "ScaledDotProductAttention.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace weft;

namespace {

template <typename T>
bool close(T a, T b, T atol = T(1e-4), T rtol = T(1e-3)) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// Sum of all entries -- our scalar "loss" for the finite-difference checks.
template <typename T>
T sum_all(const Matrix<T>& M) {
    T s = T(0);
    for (std::size_t j = 0; j < M.cols(); ++j)
        for (std::size_t i = 0; i < M.rows(); ++i)
            s += M(i, j);
    return s;
}

template <typename T>
Matrix<T> rand_mat(std::size_t r, std::size_t c, unsigned seed,
                   T lo = T(-1), T hi = T(1))
{
    Matrix<T> M(r, c);
    M.randomizeUniform(lo, hi, seed);
    return M;
}

int n_passed = 0;
int n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

} // namespace

// -- 1. Forward shape -----------------------------------------------------
void test_forward_shape() {
    std::cout << "shape:\n";
    ScaledDotProductAttention<float> a;
    auto Q = rand_mat<float>(4, 3, 1);   // d_k=4, n_q=3
    auto K = rand_mat<float>(4, 5, 2);   // d_k=4, n_k=5
    auto V = rand_mat<float>(7, 5, 3);   // d_v=7, n_k=5
    auto Y = a.forward(Q, K, V);
    check(Y.rows() == 7 && Y.cols() == 3, "output is (d_v x n_q)");
    const auto& A = a.attention();
    check(A.rows() == 5 && A.cols() == 3, "attention map is (n_k x n_q)");
}

// -- 2. Attention rows-per-column sum to 1 --------------------------------
void test_attention_normalised() {
    std::cout << "softmax-per-column sums to 1:\n";
    ScaledDotProductAttention<float> a;
    auto Q = rand_mat<float>(8, 4, 10);
    auto K = rand_mat<float>(8, 6, 11);
    auto V = rand_mat<float>(8, 6, 12);
    a.forward(Q, K, V);
    const auto& A = a.attention();
    bool all_ok = true;
    for (std::size_t j = 0; j < A.cols(); ++j) {
        float s = 0.0f;
        for (std::size_t i = 0; i < A.rows(); ++i) s += A(i, j);
        if (!close(s, 1.0f)) { all_ok = false; break; }
    }
    check(all_ok, "every column sums to 1");
}

// -- 3. One-hot keys: attention picks out the matching value -------------
// If Q has the same row pattern as K[:, j*], then softmax should
// concentrate weight on key j*, and the output equals V[:, j*].
void test_one_hot_keys() {
    std::cout << "one-hot keys recover the matching value:\n";
    ScaledDotProductAttention<float> a;
    const std::size_t d_k = 4, n_k = 3, d_v = 5;
    Matrix<float> K(d_k, n_k);
    for (std::size_t i = 0; i < d_k; ++i)
        for (std::size_t j = 0; j < n_k; ++j) K(i, j) = (i == j) ? 30.0f : 0.0f;
    // Sharp peak temperature so the softmax is effectively a hard argmax.

    Matrix<float> V(d_v, n_k);
    for (std::size_t i = 0; i < d_v; ++i)
        for (std::size_t j = 0; j < n_k; ++j) V(i, j) = static_cast<float>(j * 10 + i);

    // Query that's "the same as key 1".
    Matrix<float> Q(d_k, 1);
    for (std::size_t i = 0; i < d_k; ++i) Q(i, 0) = (i == 1) ? 30.0f : 0.0f;

    auto Y = a.forward(Q, K, V);
    bool ok = true;
    for (std::size_t i = 0; i < d_v; ++i)
        if (!close(Y(i, 0), V(i, 1), 1e-3f, 1e-3f)) ok = false;
    check(ok, "Y equals V[:, 1] when query matches key 1 sharply");
}

// -- 4. Finite-difference gradient checks (double precision) -------------
// Loss = sum(forward(Q, K, V)).  Then dQ_{i,j} (analytical) should equal
// (L(Q + eps e_{ij}) - L(Q - eps e_{ij})) / (2 eps).  Same for K, V.
void test_numerical_grads() {
    std::cout << "numerical gradient (double precision):\n";
    using D = double;
    const std::size_t d_k = 5, n_q = 3, n_k = 4, d_v = 6;
    auto Q = rand_mat<D>(d_k, n_q, 100, -0.5, 0.5);
    auto K = rand_mat<D>(d_k, n_k, 101, -0.5, 0.5);
    auto V = rand_mat<D>(d_v, n_k, 102, -0.5, 0.5);

    ScaledDotProductAttention<D> a;
    auto Y  = a.forward(Q, K, V);
    Matrix<D> dY(Y.rows(), Y.cols());
    for (std::size_t j = 0; j < dY.cols(); ++j)
        for (std::size_t i = 0; i < dY.rows(); ++i) dY(i, j) = 1.0;
    auto [dQ_an, dK_an, dV_an] = a.backward(dY);

    const D eps = 1e-5;
    auto perturb = [&](Matrix<D>& M, std::size_t i, std::size_t j, D delta) {
        M(i, j) += delta;
    };
    auto loss_for = [&](Matrix<D> q, Matrix<D> k, Matrix<D> v) {
        ScaledDotProductAttention<D> b;
        return sum_all(b.forward(q, k, v));
    };

    auto check_grad = [&](const Matrix<D>& dM_an, const char* name,
                          auto get_loss_at)
    {
        D worst = 0.0;
        for (std::size_t j = 0; j < dM_an.cols(); ++j)
            for (std::size_t i = 0; i < dM_an.rows(); ++i) {
                const D lp = get_loss_at(i, j, +eps);
                const D ln = get_loss_at(i, j, -eps);
                const D num = (lp - ln) / (2.0 * eps);
                const D err = std::fabs(num - dM_an(i, j));
                if (err > worst) worst = err;
            }
        std::cout << "  worst |num - analytic| for d" << name
                  << " = " << worst << "\n";
        check(worst < 1e-5, std::string("d") + name + " matches numerical");
    };

    check_grad(dQ_an, "Q",
        [&](std::size_t i, std::size_t j, D delta) {
            auto Qp = Q; perturb(Qp, i, j, delta);
            return loss_for(Qp, K, V);
        });
    check_grad(dK_an, "K",
        [&](std::size_t i, std::size_t j, D delta) {
            auto Kp = K; perturb(Kp, i, j, delta);
            return loss_for(Q, Kp, V);
        });
    check_grad(dV_an, "V",
        [&](std::size_t i, std::size_t j, D delta) {
            auto Vp = V; perturb(Vp, i, j, delta);
            return loss_for(Q, K, Vp);
        });
}

int main() {
    std::cout << "test_scaled_dot_product_attention\n";
    test_forward_shape();
    test_attention_normalised();
    test_one_hot_keys();
    test_numerical_grads();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
