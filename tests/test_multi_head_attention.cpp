// test_multi_head_attention.cpp  --  verify the layer interface, the
// shape contracts, and the backward pass (against finite differences in
// double precision).

#include "Matrix.h"
#include "MultiHeadAttention.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using namespace weft;

namespace {

template <typename T>
bool close(T a, T b, T atol = T(1e-4), T rtol = T(1e-3)) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}
template <typename T>
T sum_all(const Matrix<T>& M) {
    T s = T(0);
    for (std::size_t j = 0; j < M.cols(); ++j)
        for (std::size_t i = 0; i < M.rows(); ++i) s += M(i, j);
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

int n_passed = 0, n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

} // namespace

void test_shape_and_constraints() {
    std::cout << "shape + constructor constraints:\n";
    MultiHeadAttention<float> mha(8, 2);
    auto X = rand_mat<float>(8, 5, 42);
    auto Y = mha.forward(X);
    check(Y.rows() == 8 && Y.cols() == 5,
          "output shape (d_model x seq_len) matches input");

    bool caught = false;
    try { MultiHeadAttention<float> bad(8, 3); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "constructor rejects d_model not divisible by n_heads");

    caught = false;
    try { MultiHeadAttention<float> bad(8, 0); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "constructor rejects n_heads = 0");

    caught = false;
    try {
        MultiHeadAttention<float> mha2(8, 2);
        auto Xbad = rand_mat<float>(5, 3, 0);          // wrong feature count
        mha2.forward(Xbad);
    } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "forward rejects input rows != d_model");
}

void test_describe() {
    std::cout << "describe():\n";
    MultiHeadAttention<float> mha(64, 8);
    check(mha.describe() == "MultiHeadAttention(d_model=64, n_heads=8)",
          "describe string matches");
}

// dX numerical gradient: loss = sum(forward(X)), check d/dX (numerical)
// against the analytical dX returned by backward.
void test_dx_numerical() {
    std::cout << "numerical gradient w.r.t. input X (double):\n";
    using D = double;
    const std::size_t d_model = 6, n_heads = 2, seq_len = 4;
    auto X = rand_mat<D>(d_model, seq_len, 1234, -0.5, 0.5);

    MultiHeadAttention<D> mha(d_model, n_heads);
    auto Y  = mha.forward(X);
    Matrix<D> dY(Y.rows(), Y.cols());
    for (std::size_t j = 0; j < dY.cols(); ++j)
        for (std::size_t i = 0; i < dY.rows(); ++i) dY(i, j) = 1.0;
    auto dX_an = mha.backward(dY);

    const D eps = 1e-5;
    D worst = 0.0;
    for (std::size_t j = 0; j < X.cols(); ++j)
        for (std::size_t i = 0; i < X.rows(); ++i) {
            auto Xp = X; Xp(i, j) += eps;
            auto Xm = X; Xm(i, j) -= eps;
            // Each evaluation needs a FRESH layer with the same initialisation,
            // otherwise the cached forward state gets clobbered.  Use seed
            // both on Matrix to drive Dense's W init: it'll match the previous
            // mha if we rebuild with no special seeding.  Dense initialises
            // from its own internal RNG seed -- so to avoid mismatches, just
            // re-use the *same* mha object: each forward overwrites the cache,
            // and we only check dX which doesn't depend on parameter grad
            // bookkeeping.
            D lp = sum_all(mha.forward(Xp));
            D lm = sum_all(mha.forward(Xm));
            D num = (lp - lm) / (2.0 * eps);
            D err = std::fabs(num - dX_an(i, j));
            if (err > worst) worst = err;
        }
    std::cout << "  worst |num - analytic| for dX = " << worst << "\n";
    check(worst < 1e-5, "dX matches numerical");
}

// Verify save/load round-trip preserves the forward output.
void test_save_load_roundtrip() {
    std::cout << "save/load round-trip:\n";
    MultiHeadAttention<float> mha_a(8, 2);
    auto X = rand_mat<float>(8, 3, 99);
    auto Y_a = mha_a.forward(X);

    // Save mha_a's params to memory.
    std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
    mha_a.save_params(buf);

    // Build a fresh MHA, load, run.  Output should match exactly.
    MultiHeadAttention<float> mha_b(8, 2);
    auto Y_before = mha_b.forward(X);   // certainly different from Y_a
    mha_b.load_params(buf);
    auto Y_after = mha_b.forward(X);

    // After load, both should match Y_a entry-wise (modulo float order).
    bool match = (Y_a.rows() == Y_after.rows() && Y_a.cols() == Y_after.cols());
    float worst = 0.0f;
    for (std::size_t j = 0; match && j < Y_a.cols(); ++j)
        for (std::size_t i = 0; i < Y_a.rows(); ++i) {
            const float e = std::fabs(Y_a(i, j) - Y_after(i, j));
            if (e > worst) worst = e;
        }
    std::cout << "  worst |Y_a - Y_after_load| = " << worst << "\n";
    check(match && worst < 1e-6f, "Y matches after save->load on fresh layer");

    // And the freshly-built layer (pre-load) shouldn't already match Y_a:
    bool diff = false;
    for (std::size_t j = 0; !diff && j < Y_a.cols(); ++j)
        for (std::size_t i = 0; i < Y_a.rows(); ++i)
            if (std::fabs(Y_a(i, j) - Y_before(i, j)) > 1e-3f) { diff = true; break; }
    check(diff, "fresh layer's output differs from saved layer's (sanity)");
}

int main() {
    std::cout << "test_multi_head_attention\n";
    test_shape_and_constraints();
    test_describe();
    test_dx_numerical();
    test_save_load_roundtrip();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
