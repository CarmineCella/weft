// test_optimizer.cpp
//
// Tests for the Optimizer abstraction:
//   - SGD: single update is correct
//   - SGD: converges on a quadratic
//   - Adam: first-step formula is correct
//   - Adam: per-parameter state is independent
//   - Adam: converges on a quadratic
//   - Adam vs SGD on a poorly-conditioned problem (Adam should win)
//
#include "SGD.h"
#include "Adam.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
using weft::SGD;
using weft::Adam;

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
    std::cout << "weft :: Optimizer tests\n";

    // ---- 1. SGD: single update is W -= lr * dW ----
    {
        SGD<float> opt(0.1f);
        Matrix<float> W{{1, 2}, {3, 4}};
        Matrix<float> dW{{10, 20}, {30, 40}};
        opt.step(W, dW);
        check(matClose(W, Matrix<float>{{0, 0}, {0, 0}}),
              "SGD: W -= lr * dW");
    }

    // ---- 2. SGD converges on f(x) = 0.5 x^2 (gradient = x) ----
    {
        SGD<float> opt(0.1f);
        Matrix<float> x{{5.0f}};
        for (int i = 0; i < 100; ++i) {
            Matrix<float> g{{x(0, 0)}};
            opt.step(x, g);
        }
        check(std::fabs(x(0, 0)) < 0.001f, "SGD: converges to x = 0");
    }

    // ---- 3. Adam: first step is approx -lr * sign(grad) ----
    //   With m=v=0 init, step 1 gives m_hat = g, v_hat = g^2,
    //   so update = lr * g / (|g| + eps) ~= lr * sign(g).
    //   Here g = 0.5 > 0, lr = 0.1, so W goes from 1.0 to ~0.9.
    {
        Adam<float> opt(0.1f);
        Matrix<float> W{{1.0f}};
        Matrix<float> dW{{0.5f}};
        opt.step(W, dW);
        check(std::fabs(W(0, 0) - 0.9f) < 1e-4f,
              "Adam: first step is approximately -lr * sign(grad)");
    }

    // ---- 4. Adam's per-parameter state is independent ----
    {
        Adam<float> opt(0.01f);
        Matrix<float> W1{{1.0f}};
        Matrix<float> W2{{1.0f}};
        Matrix<float> dW1{{ 0.5f}};
        Matrix<float> dW2{{-0.5f}};
        opt.step(W1, dW1);
        opt.step(W2, dW2);
        check(W1(0, 0) < 1.0f && W2(0, 0) > 1.0f,
              "Adam: independent state per parameter");
    }

    // ---- 5. Adam converges on a quadratic ----
    {
        Adam<float> opt(0.1f);
        Matrix<float> x{{5.0f}};
        for (int i = 0; i < 200; ++i) {
            Matrix<float> g{{x(0, 0)}};
            opt.step(x, g);
        }
        check(std::fabs(x(0, 0)) < 0.01f, "Adam: converges to x = 0");
    }

    // ---- 6. Poorly-conditioned quadratic: Adam beats SGD ----
    //   f(x, y) = 0.5 x^2 + 50 y^2   ->   gradient = (x, 100 y)
    //   Gradient on y is 100x the gradient on x. Plain SGD with one
    //   learning rate can't be aggressive on x AND stable on y;
    //   Adam adapts per-parameter.
    {
        // SGD: lr small enough to stay stable on the y-direction
        SGD<float> sgd(0.01f);
        Matrix<float> w_sgd{{5.0f, 5.0f}};
        for (int i = 0; i < 100; ++i) {
            Matrix<float> g{{w_sgd(0, 0), 100.0f * w_sgd(0, 1)}};
            sgd.step(w_sgd, g);
        }
        float sgd_loss = 0.5f * (w_sgd(0,0)*w_sgd(0,0)
                                 + 100.f * w_sgd(0,1)*w_sgd(0,1));

        // Adam: same starting point
        Adam<float> adam(0.1f);
        Matrix<float> w_adam{{5.0f, 5.0f}};
        for (int i = 0; i < 100; ++i) {
            Matrix<float> g{{w_adam(0, 0), 100.0f * w_adam(0, 1)}};
            adam.step(w_adam, g);
        }
        float adam_loss = 0.5f * (w_adam(0,0)*w_adam(0,0)
                                  + 100.f * w_adam(0,1)*w_adam(0,1));

        std::cout << "    (after 100 steps: SGD loss = " << sgd_loss
                  << ", Adam loss = " << adam_loss << ")\n";
        check(adam_loss < sgd_loss * 0.5f,
              "Adam beats SGD by >2x on a poorly-conditioned quadratic");
    }

    std::cout << "\n" << (g_run - g_failed) << " / " << g_run
              << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
