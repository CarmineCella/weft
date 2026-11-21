// test_embedding.cpp  --  verify Embedding: forward is a column lookup,
// backward scatters the gradient into the looked-up columns (adding when
// a token repeats), the gradient matches finite differences, training
// moves only the columns that were used, and save/load round-trips.

#include "Adam.h"
#include "Embedding.h"
#include "Matrix.h"
#include "SGD.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace weft;

namespace {

int n_passed = 0, n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

} // namespace

void test_forward_is_lookup() {
    std::cout << "forward is a column lookup:\n";
    Embedding<float> emb(/*vocab=*/7, /*d_model=*/4, /*seed=*/3);
    const std::vector<int> tokens = {5, 0, 5, 2};
    auto Y = emb.forward(tokens);
    check(Y.rows() == 4 && Y.cols() == 4, "output is (d_model x seq_len)");
    bool same = true;
    for (std::size_t t = 0; t < tokens.size(); ++t)
        for (std::size_t d = 0; d < 4; ++d)
            if (Y(d, t) != emb.W()(d, std::size_t(tokens[t]))) same = false;
    check(same, "column t of the output is column tokens[t] of the table");

    bool caught = false;
    try { emb.forward({1, 7}); } catch (const std::out_of_range&) { caught = true; }
    check(caught, "a token id >= vocab_size is rejected");
    caught = false;
    try { emb.forward({-1}); } catch (const std::out_of_range&) { caught = true; }
    check(caught, "a negative token id is rejected");
}

// Token 5 appears twice, so its column must receive the SUM of the two
// upstream gradient columns; unused tokens must receive exactly zero.
void test_backward_scatter_add() {
    std::cout << "backward scatters and adds:\n";
    Embedding<double> emb(7, 3, 4);
    const std::vector<int> tokens = {5, 0, 5, 2};
    emb.forward(tokens);
    Matrix<double> dY(3, 4);
    dY.randomizeUniform(-1.0, 1.0, 9);
    emb.backward(dY);
    const auto& dW = emb.dW();

    bool repeated = true, single = true, unused = true;
    for (std::size_t d = 0; d < 3; ++d) {
        if (std::fabs(dW(d, 5) - (dY(d, 0) + dY(d, 2))) > 1e-15) repeated = false;
        if (dW(d, 0) != dY(d, 1) || dW(d, 2) != dY(d, 3)) single = false;
        for (std::size_t k : {1u, 3u, 4u, 6u})
            if (dW(d, k) != 0.0) unused = false;
    }
    check(repeated, "a repeated token gets the sum of its gradient columns");
    check(single,   "a token used once gets its own gradient column");
    check(unused,   "tokens not in the sequence get exactly zero");

    // A second backward must not accumulate on top of the first.
    emb.backward(dY);
    bool reset = true;
    for (std::size_t d = 0; d < 3; ++d)
        if (std::fabs(emb.dW()(d, 5) - (dY(d, 0) + dY(d, 2))) > 1e-15) reset = false;
    check(reset, "dW is reset at every backward");

    bool caught = false;
    try { emb.backward(Matrix<double>(3, 5)); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "backward rejects a gradient of the wrong shape");
}

// Loss = sum(Wt .* forward(tokens)).  Perturb the table through an SGD
// step (the table has no setter): a step of size lr with gradient G
// changes W by -lr * G, so we can nudge one entry at a time.
void test_numerical_gradient() {
    std::cout << "numerical gradient (double precision):\n";
    const std::size_t vocab = 6, d = 3;
    const std::vector<int> tokens = {4, 1, 4, 4, 0};
    Matrix<double> Wt(d, tokens.size());
    Wt.randomizeUniform(-1.0, 1.0, 11);

    auto loss_of = [&](Embedding<double>& e) {
        auto Y = e.forward(tokens);
        double s = 0.0;
        for (std::size_t j = 0; j < Y.cols(); ++j)
            for (std::size_t i = 0; i < Y.rows(); ++i) s += Wt(i, j) * Y(i, j);
        return s;
    };

    Embedding<double> emb(vocab, d, 5);
    loss_of(emb);
    emb.backward(Wt);
    const Matrix<double> dW = emb.dW();

    const double eps = 1e-6;
    double worst = 0.0;
    for (std::size_t k = 0; k < vocab; ++k)
        for (std::size_t i = 0; i < d; ++i) {
            // Nudge W(i, k) by +eps and -eps using a hand-built "gradient".
            auto nudged = [&](double delta) {
                Embedding<double> e(vocab, d, 5);          // same seed, same table
                e.forward({int(k)});
                Matrix<double> g(d, 1, 0.0);
                g(i, 0) = -delta;                          // SGD: W -= lr * g
                e.backward(g);
                SGD<double> one(1.0);
                e.update(one);
                return loss_of(e);
            };
            const double num = (nudged(+eps) - nudged(-eps)) / (2 * eps);
            worst = std::max(worst, std::fabs(num - dW(i, k)));
        }
    std::cout << "  worst |num - analytic| for dW = " << worst << "\n";
    check(worst < 1e-8, "dW matches numerical");
}

// Training a table so that one token's vector matches a target must move
// that token's column and leave every other column untouched.
void test_training_moves_only_used_columns() {
    std::cout << "training moves only the used columns:\n";
    Embedding<float> emb(5, 4, 6);
    const Matrix<float> before = emb.W();
    Matrix<float> target(4, 1);
    for (std::size_t i = 0; i < 4; ++i) target(i, 0) = 0.5f * float(i) - 0.7f;

    Adam<float> opt(0.05f);
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 300; ++step) {
        auto Y = emb.forward({3});
        Matrix<float> dY(4, 1);
        float loss = 0.0f;
        for (std::size_t i = 0; i < 4; ++i) {
            const float e = Y(i, 0) - target(i, 0);
            loss += 0.5f * e * e;
            dY(i, 0) = e;
        }
        if (step == 0) first = loss;
        last = loss;
        emb.backward(dY);
        emb.update(opt);
    }
    std::cout << "  loss " << first << " -> " << last << "\n";
    check(last < 1e-4f * std::max(first, 1e-3f), "the used token's vector reaches the target");

    bool others_same = true;
    for (std::size_t k = 0; k < 5; ++k) {
        if (k == 3) continue;
        for (std::size_t i = 0; i < 4; ++i)
            if (emb.W()(i, k) != before(i, k)) others_same = false;
    }
    check(others_same, "every other token's vector is unchanged");
}

void test_save_load() {
    std::cout << "save/load round-trip:\n";
    Embedding<float> a(9, 5, 1), b(9, 5, 2);
    std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
    a.save_params(buf);
    b.load_params(buf);
    bool same = true;
    for (std::size_t k = 0; k < 9; ++k)
        for (std::size_t i = 0; i < 5; ++i)
            if (a.W()(i, k) != b.W()(i, k)) same = false;
    check(same, "the table survives save -> load");

    bool caught = false;
    try {
        std::stringstream buf2(std::ios::in | std::ios::out | std::ios::binary);
        a.save_params(buf2);
        Embedding<float> wrong(8, 5, 1);
        wrong.load_params(buf2);
    } catch (const std::runtime_error&) { caught = true; }
    check(caught, "load rejects a table of a different shape");
}

int main() {
    std::cout << "test_embedding\n";
    test_forward_is_lookup();
    test_backward_scatter_add();
    test_numerical_gradient();
    test_training_moves_only_used_columns();
    test_save_load();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
