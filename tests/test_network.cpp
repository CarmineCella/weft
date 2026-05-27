// test_network.cpp
//
// Tests for the Network class. The big one is the full-pipeline numerical
// gradient check: we build a complete classifier, perturb each weight in
// the first Dense layer, confirm the analytical gradient produced by the
// chain matches the finite difference. If that passes, every component in
// the library is correctly wired together.
//
//   1. Forward composition matches a manual chain.
//   2. Full pipeline: numerical dW vs analytical dW on the first Dense.
//   3. Training run: SGD reduces a 3-class synthetic classification loss.
//
#include "Network.h"
#include "Dense.h"
#include "ReLU.h"
#include "Softmax.h"
#include "CrossEntropy.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
using weft::Network;
using weft::Dense;
using weft::ReLU;
using weft::Softmax;
using weft::CrossEntropy;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}
static bool matClose(const Matrix<float>& A, const Matrix<float>& B,
                     float eps = 1e-4f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            if (!close(A(i, j), B(i, j), eps)) return false;
    return true;
}

int main() {
    std::cout << "weft :: Network tests\n";

    // ---- 1. Forward composition matches a hand-chained sequence ----
    {
        Network<float> net;
        auto& d1 = net.add<Dense>(2, 3);
        net.add<ReLU>();
        auto& d2 = net.add<Dense>(3, 2);

        d1.W() = Matrix<float>{{ 1, -1}, { 2,  0}, {-1,  1}};
        d1.b() = Matrix<float>{{0.1f}, {0.2f}, {0.3f}};
        d2.W() = Matrix<float>{{ 1,  1, 1}, { 1, -1, 0}};
        d2.b() = Matrix<float>{{0.0f}, {0.0f}};

        Matrix<float> X{{1, 0}, {1, -1}};
        Matrix<float> Y_net = net.forward(X);

        // Manual chain with separate instances using the same weights.
        Dense<float> a(2, 3);  a.W() = d1.W();  a.b() = d1.b();
        ReLU<float>  r;
        Dense<float> b(3, 2);  b.W() = d2.W();  b.b() = d2.b();
        Matrix<float> Y_manual = b.forward(r.forward(a.forward(X)));

        check(matClose(Y_net, Y_manual),
              "forward: Network output == manual chain");
    }

    // ---- 2. Full pipeline numerical gradient check ----
    //   Dense -> ReLU -> Dense -> Softmax -> CrossEntropy
    //   Perturb each entry of the first Dense's W, confirm the analytical
    //   dW (from the full backward pass) matches (L+ - L-) / (2 eps).
    {
        Network<float> net;
        auto& d1 = net.add<Dense>(3, 5);    // first Dense -- the one we'll inspect
        net.add<ReLU>();
        net.add<Dense>(5, 4);
        net.add<Softmax>();
        CrossEntropy<float> ce;

        Matrix<float> X(3, 4);
        X.randomizeUniform(-1.f, 1.f, /*seed=*/42);

        // One-hot targets across 4 examples, 4 classes.
        Matrix<float> Target{{1, 0, 0, 0},
                             {0, 1, 0, 0},
                             {0, 0, 1, 0},
                             {0, 0, 0, 1}};

        // Analytical: one full forward + backward.
        Matrix<float> S  = net.forward(X);
        ce.forward(S, Target);
        Matrix<float> dS = ce.backward();
        net.backward(dS);

        // For each W[i,j] of the first Dense, compute (L(W+eps)-L(W-eps))/(2eps).
        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < d1.W().rows() && ok; ++i) {
            for (std::size_t j = 0; j < d1.W().cols() && ok; ++j) {
                float orig = d1.W()(i, j);

                d1.W()(i, j) = orig + eps;
                float L_plus  = ce.forward(net.forward(X), Target);
                d1.W()(i, j) = orig - eps;
                float L_minus = ce.forward(net.forward(X), Target);
                d1.W()(i, j) = orig;

                float numerical  = (L_plus - L_minus) / (2 * eps);
                float analytical = d1.dW()(i, j);
                if (!close(numerical, analytical, 1e-2f)) ok = false;
            }
        }
        check(ok, "full pipeline: numerical dW == analytical dW on first Dense");
    }

    // ---- 3. End-to-end training: SGD drives cross-entropy loss down ----
    //   Small synthetic 3-class problem. Inputs: 4D random points labelled
    //   by a fixed linear rule. The network has to recover separating
    //   boundaries through Dense -> ReLU -> Dense -> Softmax.
    {
        Network<float> net;
        net.add<Dense>(4, 8);
        net.add<ReLU>();
        net.add<Dense>(8, 3);
        net.add<Softmax>();
        CrossEntropy<float> ce;

        // 12 training examples, 4 features, 3 classes (4 examples per class).
        Matrix<float> X(4, 12);
        X.randomizeUniform(-1.f, 1.f, /*seed=*/7);

        // Targets: class k for the k-th group of 4 columns.
        Matrix<float> Target(3, 12, 0.f);
        for (std::size_t j = 0; j < 12; ++j) Target(j / 4, j) = 1.f;

        // Train for 200 steps.
        float L0 = ce.forward(net.forward(X), Target);
        for (int step = 0; step < 200; ++step) {
            ce.forward(net.forward(X), Target);
            net.backward(ce.backward());
            net.update(0.2f);
        }
        float L_final = ce.forward(net.forward(X), Target);

        // We don't insist on a tight threshold -- just that SGD makes
        // meaningful progress on a small problem.
        check(L_final < L0 * 0.3f,
              "training: SGD reduces cross-entropy by >3x in 200 steps");
    }

    std::cout << "\n" << (g_run - g_failed) << " / " << g_run
              << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
