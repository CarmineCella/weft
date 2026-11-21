// test_transformer_encoder_block.cpp  --  verify the pre-LN encoder
// block: shapes, the input gradient against finite differences (with and
// without the causal mask), the causal "no leak" property, that a
// training step actually lowers a loss, and the save/load round-trip.

#include "Adam.h"
#include "Matrix.h"
#include "MSE.h"
#include "TransformerEncoderBlock.h"

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

// Scalar loss  L = sum(Wt .* Y), so every output entry matters
// differently and the gradient fed to backward is simply Wt.
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

// Worst |numerical - analytic| over every entry of dX.
double dx_error(TransformerEncoderBlock<double>& blk,
                const Matrix<double>& X, const Matrix<double>& Wt)
{
    blk.forward(X);
    auto dX = blk.backward(Wt);
    // Step 1e-6, smaller than the 1e-5 used for single layers.  The block
    // is a long chain, so the finite difference's own truncation error
    // (which grows as eps^2) reaches ~1e-7 at 1e-5 for some random
    // initialisations; at 1e-6 it stays near 1e-9.
    const double eps = 1e-6;
    double worst = 0.0;
    for (std::size_t j = 0; j < X.cols(); ++j)
        for (std::size_t i = 0; i < X.rows(); ++i) {
            auto Xp = X, Xm = X;
            Xp(i, j) += eps; Xm(i, j) -= eps;
            const double num = (weighted_sum(Wt, blk.forward(Xp))
                              - weighted_sum(Wt, blk.forward(Xm))) / (2.0 * eps);
            worst = std::max(worst, std::fabs(num - dX(i, j)));
        }
    return worst;
}

} // namespace

void test_shape_and_describe() {
    std::cout << "shape and describe():\n";
    TransformerEncoderBlock<float> blk(8, 2);
    auto X = rand_mat<float>(8, 5, 1);
    auto Y = blk.forward(X);
    check(Y.rows() == 8 && Y.cols() == 5, "output shape equals input shape");
    check(blk.d_ff() == 32, "d_ff defaults to 4 * d_model");
    check(blk.describe() ==
          "TransformerEncoderBlock(d_model=8, n_heads=2, d_ff=32)",
          "describe string matches");

    TransformerEncoderBlock<float> small(8, 4, 12);
    check(small.d_ff() == 12, "explicit d_ff is respected");

    bool caught = false;
    try { TransformerEncoderBlock<float> bad(8, 3); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "constructor rejects d_model not divisible by n_heads");
}

void test_dx_numerical() {
    std::cout << "numerical gradient w.r.t. input X (double):\n";
    const std::size_t d_model = 6, n_heads = 2, seq_len = 4;
    auto X  = rand_mat<double>(d_model, seq_len, 21, -0.5, 0.5);
    auto Wt = rand_mat<double>(d_model, seq_len, 22);

    TransformerEncoderBlock<double> blk(d_model, n_heads, 10);
    const double e = dx_error(blk, X, Wt);
    std::cout << "  worst |num - analytic| for dX = " << e << "\n";
    check(e < 1e-7, "dX matches numerical (bidirectional)");

    blk.set_causal(true);
    const double ec = dx_error(blk, X, Wt);
    std::cout << "  worst |num - analytic| for dX (causal) = " << ec << "\n";
    check(ec < 1e-7, "dX matches numerical (causal)");
}

// With the causal mask on, output column j depends only on input columns
// 0..j.  Without it, the change reaches the earlier columns.
void test_causal_no_leak() {
    std::cout << "causal mode (no leak from the future):\n";
    const std::size_t d_model = 8, n_heads = 2, seq_len = 6, t = 3;
    auto X = rand_mat<double>(d_model, seq_len, 31, -0.5, 0.5);
    // The change must differ from feature to feature: LayerNorm removes
    // any constant added to a whole column, so a uniform shift would be
    // invisible to the block and the test would prove nothing.
    auto X2 = X;
    for (std::size_t i = 0; i < d_model; ++i)
        X2(i, t) += (i % 2 ? 0.9 : -0.6) * (1.0 + i);

    TransformerEncoderBlock<double> blk(d_model, n_heads);
    check(!blk.causal(), "block starts bidirectional");

    auto leak_before_t = [&]() {
        auto Y0 = blk.forward(X);
        auto Y1 = blk.forward(X2);
        double e = 0.0;
        for (std::size_t j = 0; j < t; ++j)
            for (std::size_t i = 0; i < d_model; ++i)
                e = std::max(e, std::fabs(Y1(i, j) - Y0(i, j)));
        return e;
    };

    check(leak_before_t() > 1e-4,
          "bidirectional: a later token changes earlier outputs (sanity)");
    blk.set_causal(true);
    check(blk.causal(), "set_causal(true) is remembered");
    check(leak_before_t() == 0.0,
          "causal: a later token leaves earlier outputs bit-identical");
}

// The parameter gradients are not checked entry by entry here (Dense and
// LayerNorm have their own tests); instead we check that they are usable:
// a few Adam steps on a fixed regression target must lower the loss.
void test_training_reduces_loss() {
    std::cout << "training lowers the loss:\n";
    const std::size_t d_model = 8, n_heads = 2, seq_len = 6;
    auto X      = rand_mat<float>(d_model, seq_len, 41);
    auto Target = rand_mat<float>(d_model, seq_len, 42);

    TransformerEncoderBlock<float> blk(d_model, n_heads);
    MSE<float>  mse;
    Adam<float> opt(1e-2f);

    const float first = mse.forward(blk.forward(X), Target);
    float last = first;
    for (int step = 0; step < 200; ++step) {
        last = mse.forward(blk.forward(X), Target);
        blk.backward(mse.backward());
        blk.update(opt);
    }
    std::cout << "  loss " << first << " -> " << last << "\n";
    check(last < 0.2f * first, "loss falls by more than 5x in 200 Adam steps");
}

void test_save_load_roundtrip() {
    std::cout << "save/load round-trip:\n";
    TransformerEncoderBlock<float> a(8, 2), b(8, 2);
    auto X = rand_mat<float>(8, 4, 51);
    auto Ya = a.forward(X);
    auto Yb_before = b.forward(X);

    std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
    a.save_params(buf);
    b.load_params(buf);
    auto Yb = b.forward(X);

    float worst = 0.0f, differed = 0.0f;
    for (std::size_t j = 0; j < Ya.cols(); ++j)
        for (std::size_t i = 0; i < Ya.rows(); ++i) {
            worst    = std::max(worst,    std::fabs(Ya(i, j) - Yb(i, j)));
            differed = std::max(differed, std::fabs(Ya(i, j) - Yb_before(i, j)));
        }
    check(worst == 0.0f, "outputs are identical after save -> load");
    check(differed > 1e-3f, "the two blocks differed before loading (sanity)");
}

int main() {
    std::cout << "test_transformer_encoder_block\n";
    test_shape_and_describe();
    test_dx_numerical();
    test_causal_no_leak();
    test_training_reduces_loss();
    test_save_load_roundtrip();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
