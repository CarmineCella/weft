#include "Random.h"
#include "Conv2D.h"
#include "Dense.h"
#include "Dropout.h"
#include "Embedding.h"
#include "Softmax.h"
#include "TokenTransformer.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static int failed = 0;
static void check(bool ok, const std::string& name) {
    std::cout << (ok ? "[ ok ] " : "[FAIL] ") << name << '\n';
    if (!ok) ++failed;
}
template <typename F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
template <typename T> bool same(const weft::Matrix<T>& a, const weft::Matrix<T>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a.data()[i] != b.data()[i]) return false;
    return true;
}
int main() {
    using namespace weft;
    Matrix<double> x(4, 20, 1.0);
    set_seed(123);
    Dense<double> first(4, 4), second(4, 4);
    Dropout<double> dropout(0.3);
    Embedding<double> embedding(7, 4);
    const auto weights = first.W(), other_weights = second.W();
    const auto mask_output = dropout.forward(x), table = embedding.W();
    set_seed(123);
    Dense<double> again(4, 4), again_second(4, 4);
    Dropout<double> again_dropout(0.3);
    Embedding<double> again_embedding(7, 4);
    check(same(weights, again.W()) && same(other_weights, again_second.W()),
          "master seed reproduces distinct dense initializations");
    check(!same(weights, other_weights), "separate layers receive independent streams");
    check(same(mask_output, again_dropout.forward(x)), "master seed reproduces dropout masks");
    check(same(table, again_embedding.W()), "master seed reproduces embeddings");
    Dense<double> explicit_a(4, 4, 19), explicit_b(4, 4, 19);
    check(same(explicit_a.W(), explicit_b.W()), "explicit seed remains available");
    set_seed(42);
    TokenTransformer<double> a(8, 4, 2, 1, 8, 8);
    const auto p = a.forward({1, 2, 3});
    set_seed(42);
    TokenTransformer<double> b(8, 4, 2, 1, 8, 8);
    check(same(p, b.forward({1, 2, 3})), "master seed reproduces a complete transformer");
    set_seed(7);
    Conv2D<double> ca(1, 2, 3);
    Tensor4D<double> image(1, 1, 4, 4, 1.0);
    const auto ya = ca.forward(image);
    set_seed(7);
    Conv2D<double> cb(1, 2, 3);
    const auto yb = cb.forward(image);
    bool conv_same = true;
    for (std::size_t i = 0; i < ya.size(); ++i) conv_same &= ya.data()[i] == yb.data()[i];
    check(conv_same, "master seed reproduces convolution kernels");
    check(throws([] { Dropout<double> d(-0.1); }) &&
          throws([] { Dropout<double> d(1.0); }) &&
          throws([] { Dropout<double> d(std::numeric_limits<double>::quiet_NaN()); }),
          "invalid dropout probabilities are rejected");
    Softmax<double> sm;
    check(throws([&] { sm.forward(Matrix<double>(0, 1)); }), "empty softmax is rejected");
    sm.forward(Matrix<double>(2, 1, 0.0));
    check(throws([&] { sm.backward(Matrix<double>(1, 1)); }), "softmax rejects wrong gradient shape");
    check(throws([&] { first.backward(Matrix<double>(4, 20)); }), "dense backward requires forward");
    return failed ? 1 : 0;
}
