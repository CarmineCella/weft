#include "SoftmaxCrossEntropy.h"
#include "Dense.h"
#include "Network.h"
#include "SGD.h"

#include <cmath>
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
template <typename T> void extreme_logits() {
    weft::SoftmaxCrossEntropy<T> loss;
    weft::Matrix<T> z{{T(50)}, {T(-50)}}, target{{T(0)}, {T(1)}};
    const T value = loss.forward(z, target);
    auto g = loss.backward();
    check(std::abs(value - T(100)) < T(1e-5) &&
          std::abs(g(0, 0) - T(1)) < T(1e-6) &&
          std::abs(g(1, 0) + T(1)) < T(1e-6),
          "confident wrong prediction has loss 100 and gradient (1,-1)");
    z(0, 0) = T(1000); z(1, 0) = T(-1000);
    check(std::abs(loss.forward(z, target) - T(2000)) < T(1e-4) &&
          loss.backward()(1, 0) == T(-1), "underflow does not erase the correction");
    target(0, 0) = T(1); target(1, 0) = T(0);
    check(loss.forward(z, target) == T(0), "confident correct prediction stays finite");
}
int main() {
    using weft::Matrix;
    using weft::SoftmaxCrossEntropy;
    extreme_logits<float>();
    extreme_logits<double>();
    SoftmaxCrossEntropy<double> loss;
    Matrix<double> z{{1.0, -0.4}, {0.2, 0.8}, {-0.5, 0.3}};
    Matrix<double> target{{0.7, 0.1}, {0.2, 0.3}, {0.1, 0.6}};
    const double baseline = loss.forward(z, target);
    const auto gradient = loss.backward();
    bool gradient_ok = true;
    for (std::size_t i = 0; i < z.rows(); ++i)
        for (std::size_t n = 0; n < z.cols(); ++n) {
            const double original = z(i, n), step = 1e-5;
            z(i, n) = original + step;
            const double plus = loss.forward(z, target);
            z(i, n) = original - step;
            const double minus = loss.forward(z, target);
            z(i, n) = original;
            gradient_ok &= std::abs((plus - minus) / (2 * step) - gradient(i, n)) < 1e-8;
        }
    check(gradient_ok, "soft-target batched gradient agrees with finite differences");
    auto shifted = z;
    for (std::size_t i = 0; i < z.rows(); ++i)
        for (std::size_t n = 0; n < z.cols(); ++n) shifted(i, n) += 1000000.0;
    check(std::abs(loss.forward(shifted, target) - baseline) < 1e-9,
          "large common logit shift leaves the objective unchanged");
    double separate = 0;
    for (std::size_t n = 0; n < z.cols(); ++n) {
        Matrix<double> a(z.rows(), 1), b(z.rows(), 1);
        for (std::size_t i = 0; i < z.rows(); ++i) { a(i, 0) = z(i, n); b(i, 0) = target(i, n); }
        separate += loss.forward(a, b);
    }
    check(std::abs(separate / z.cols() - baseline) < 1e-12, "batch loss is an example mean");
    SoftmaxCrossEntropy<double> fresh;
    check(throws([&] { fresh.backward(); }) && throws([&] { fresh.probabilities(); }),
          "backward/probabilities require a successful forward");
    check(throws([&] { loss.forward(Matrix<double>(), Matrix<double>()); }),
          "empty batches are rejected");
    auto bad = target; bad(0, 0) = -0.1;
    check(throws([&] { loss.forward(z, bad); }), "negative target is rejected");
    bad = target; bad(0, 0) += 0.5;
    check(throws([&] { loss.forward(z, bad); }), "unnormalized target is rejected");
    bad = z; bad(0, 0) = std::numeric_limits<double>::infinity();
    check(throws([&] { loss.forward(bad, target); }), "non-finite logits are rejected");

    weft::Network<double> net;
    auto& dense = net.add<weft::Dense>(1, 2, 1);
    dense.W()(0, 0) = 50; dense.W()(1, 0) = -50;
    Matrix<double> x{{1.0}}, t{{0.0}, {1.0}};
    const double before = loss.forward(net.forward(x), t);
    net.backward(loss.backward());
    weft::SGD<double> optimizer(0.5);
    net.update(optimizer);
    check(loss.forward(net.forward(x), t) < before - 1.0,
          "a full training step corrects a saturated wrong classifier");
    return failed ? 1 : 0;
}
