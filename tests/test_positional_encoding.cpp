// test_positional_encoding.cpp  --  verify SinusoidalPositionalEncoding:
// the table matches the formula, positions are distinguishable, a shift
// in position is a fixed rotation of each (sin, cos) pair, and the layer
// adds the table in forward and passes gradients through in backward.

#include "Matrix.h"
#include "SinusoidalPositionalEncoding.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace weft;

namespace {

int n_passed = 0, n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

} // namespace

// PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(same angle).
void test_table_matches_formula() {
    std::cout << "table matches the formula:\n";
    const std::size_t d = 8, L = 50;
    SinusoidalPositionalEncoding<double> pe(d, L);
    const auto& P = pe.positional_encoding();
    check(P.rows() == d && P.cols() == L, "table is (d_model x max_seq_len)");

    double worst = 0.0;
    for (std::size_t pos = 0; pos < L; ++pos)
        for (std::size_t i = 0; i < d; i += 2) {
            const double angle = pos / std::pow(10000.0, double(i) / d);
            worst = std::max(worst, std::fabs(P(i, pos)     - std::sin(angle)));
            worst = std::max(worst, std::fabs(P(i + 1, pos) - std::cos(angle)));
        }
    check(worst < 1e-12, "every entry equals sin/cos of the stated angle");

    bool pos0 = true;
    for (std::size_t i = 0; i < d; i += 2)
        if (P(i, 0) != 0.0 || P(i + 1, 0) != 1.0) pos0 = false;
    check(pos0, "position 0 is (0, 1, 0, 1, ...)");
}

// No two positions share an encoding.
void test_positions_distinct() {
    std::cout << "positions are distinct:\n";
    const std::size_t d = 16, L = 200;
    SinusoidalPositionalEncoding<double> pe(d, L);
    const auto& P = pe.positional_encoding();
    double closest = 1e9;
    for (std::size_t a = 0; a < L; ++a)
        for (std::size_t b = a + 1; b < L; ++b) {
            double dist2 = 0.0;
            for (std::size_t i = 0; i < d; ++i)
                dist2 += (P(i, a) - P(i, b)) * (P(i, a) - P(i, b));
            closest = std::min(closest, std::sqrt(dist2));
        }
    std::cout << "  smallest distance between two positions = " << closest << "\n";
    check(closest > 1e-2, "all 200 positions are well separated");
}

// The "relative offsets are linear" property: PE(pos + k) is obtained
// from PE(pos) by rotating each (sin, cos) pair by the angle k * w_i,
// and that rotation does not depend on pos.
void test_shift_is_rotation() {
    std::cout << "a shift in position is a fixed rotation:\n";
    const std::size_t d = 8, L = 64, k = 5;
    SinusoidalPositionalEncoding<double> pe(d, L);
    const auto& P = pe.positional_encoding();
    double worst = 0.0;
    for (std::size_t pos = 0; pos + k < L; ++pos)
        for (std::size_t i = 0; i < d; i += 2) {
            const double w = 1.0 / std::pow(10000.0, double(i) / d);
            const double c = std::cos(k * w), s = std::sin(k * w);
            const double sin_pred =  c * P(i, pos) + s * P(i + 1, pos);
            const double cos_pred = -s * P(i, pos) + c * P(i + 1, pos);
            worst = std::max(worst, std::fabs(sin_pred - P(i, pos + k)));
            worst = std::max(worst, std::fabs(cos_pred - P(i + 1, pos + k)));
        }
    check(worst < 1e-12, "PE(pos + 5) = R(5) PE(pos) for every pos");
}

void test_forward_backward() {
    std::cout << "forward adds, backward passes through:\n";
    const std::size_t d = 6, L = 10, n = 4;
    SinusoidalPositionalEncoding<float> pe(d, L);
    Matrix<float> X(d, n);
    X.randomizeUniform(-1.0f, 1.0f, 5);
    auto Y = pe.forward(X);
    const auto& P = pe.positional_encoding();
    bool ok = (Y.rows() == d && Y.cols() == n);
    for (std::size_t j = 0; ok && j < n; ++j)
        for (std::size_t i = 0; i < d; ++i)
            if (Y(i, j) != X(i, j) + P(i, j)) { ok = false; break; }
    check(ok, "Y = X + PE[:, :seq_len]");

    Matrix<float> dY(d, n);
    dY.randomizeUniform(-1.0f, 1.0f, 6);
    auto dX = pe.backward(dY);
    bool same = true;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < d; ++i)
            if (dX(i, j) != dY(i, j)) same = false;
    check(same, "dX = dY");
}

void test_odd_d_model_and_errors() {
    std::cout << "odd d_model and argument checks:\n";
    SinusoidalPositionalEncoding<double> odd(5, 8);
    const auto& P = odd.positional_encoding();
    // Last row has no cosine partner; it must still be the sine of its pair.
    const double angle = 3.0 / std::pow(10000.0, 4.0 / 5.0);
    check(std::fabs(P(4, 3) - std::sin(angle)) < 1e-12,
          "odd d_model: the unpaired last row is a sine");

    SinusoidalPositionalEncoding<float> pe(4, 3);
    bool caught = false;
    try { pe.forward(Matrix<float>(4, 4)); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "forward rejects a sequence longer than max_seq_len");

    caught = false;
    try { pe.forward(Matrix<float>(5, 2)); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "forward rejects input rows != d_model");

    check(pe.describe() ==
          "SinusoidalPositionalEncoding(d_model=4, max_seq_len=3)",
          "describe string matches");
}

int main() {
    std::cout << "test_positional_encoding\n";
    test_table_matches_formula();
    test_positions_distinct();
    test_shift_is_rotation();
    test_forward_backward();
    test_odd_d_model_and_errors();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
