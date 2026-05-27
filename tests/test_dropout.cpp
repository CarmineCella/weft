// test_dropout.cpp
//
// Tests for the Dropout layer and Network::train()/eval() propagation.
//
#include "Dense.h"
#include "Dropout.h"
#include "Matrix.h"
#include "Network.h"
#include "ReLU.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Dense;
using weft::Dropout;
using weft::Matrix;
using weft::Network;
using weft::ReLU;

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
    std::cout << "weft :: Dropout tests\n";

    // ---- 1. Eval mode: forward is identity ----
    {
        Dropout<float> drop(0.5f, /*seed=*/0);
        drop.set_training(false);
        Matrix<float> X{{1, 2, 3}, {4, 5, 6}};
        Matrix<float> Y = drop.forward(X);
        check(matClose(Y, X), "eval mode: forward is identity");
    }

    // ---- 2. Eval mode: backward is identity ----
    {
        Dropout<float> drop(0.5f, /*seed=*/0);
        drop.set_training(false);
        Matrix<float> X{{1, 2}, {3, 4}};
        drop.forward(X);
        Matrix<float> dY{{10, 20}, {30, 40}};
        Matrix<float> dX = drop.backward(dY);
        check(matClose(dX, dY), "eval mode: backward is identity");
    }

    // ---- 3. Training mode, rate = 0: forward is identity ----
    {
        Dropout<float> drop(0.0f, /*seed=*/0);  // training is default
        Matrix<float> X{{1, 2, 3}, {4, 5, 6}};
        Matrix<float> Y = drop.forward(X);
        check(matClose(Y, X), "training rate=0: forward is identity");
    }

    // ---- 4. Training mode, rate = 0.5:
    //         every entry is either 0 or 2 (= 1 / (1 - 0.5)),
    //         and the empirical mean over a large matrix approaches 1.
    {
        Dropout<float> drop(0.5f, /*seed=*/42);
        Matrix<float> X(100, 100, 1.0f);
        Matrix<float> Y = drop.forward(X);

        bool values_ok = true;
        float total    = 0;
        std::size_t zeros = 0;
        for (std::size_t i = 0; i < Y.rows(); ++i)
            for (std::size_t j = 0; j < Y.cols(); ++j) {
                float y = Y(i, j);
                if (!(close(y, 0.0f) || close(y, 2.0f))) values_ok = false;
                if (close(y, 0.0f)) ++zeros;
                total += y;
            }
        check(values_ok, "training rate=0.5: every value is 0 or 2");

        // With 10,000 samples at rate=0.5, the fraction of zeros is roughly
        // 0.5 ± 0.005.  Loose tolerance keeps the test rock solid.
        float zero_frac = static_cast<float>(zeros) / (Y.rows() * Y.cols());
        check(std::fabs(zero_frac - 0.5f) < 0.03f,
              "training rate=0.5: zero fraction ~= 0.5");

        float mean = total / (Y.rows() * Y.cols());
        check(std::fabs(mean - 1.0f) < 0.05f,
              "training rate=0.5: empirical mean ~= input mean (1.0)");
    }

    // ---- 5. Backward uses the same mask as forward.
    //         With X = all ones, Y(i,j) = mask(i,j) (which is 0 or scale).
    //         So  dX(i,j) = Y(i,j) * dY(i,j).
    {
        Dropout<float> drop(0.5f, /*seed=*/42);
        Matrix<float> X(10, 10, 1.0f);
        Matrix<float> Y = drop.forward(X);

        // Vary dY so the test catches "backward returns Y instead of dY*Y"
        // or "backward returns dY instead of Y*dY".
        Matrix<float> dY(10, 10, 0.0f);
        for (std::size_t i = 0; i < 10; ++i)
            for (std::size_t j = 0; j < 10; ++j)
                dY(i, j) = static_cast<float>(i + j) * 0.1f;

        Matrix<float> dX = drop.backward(dY);

        bool ok = true;
        for (std::size_t i = 0; i < 10 && ok; ++i)
            for (std::size_t j = 0; j < 10 && ok; ++j) {
                float expected = Y(i, j) * dY(i, j);
                if (!close(dX(i, j), expected)) ok = false;
            }
        check(ok, "backward: dX = mask * dY  (same mask as forward)");
    }

    // ---- 6. Network::eval() / train() propagates to Dropout layers ----
    {
        Network<float> net;
        net.add<Dense>(4, 8);
        net.add<ReLU>();
        auto& drop = net.add<Dropout>(0.5f, /*seed=*/123);
        net.add<Dense>(8, 3);

        // After eval(), forward through Dropout should be identity in
        // terms of the mask never being applied.  We can detect this by
        // checking the dropout layer's training() flag directly.
        net.eval();
        check(drop.training() == false, "Network::eval(): Dropout sees training=false");

        net.train();
        check(drop.training() == true,  "Network::train(): Dropout sees training=true");
    }

    // ---- 7. End-to-end: eval mode of Dropout-containing network is
    //         deterministic across forward calls.
    {
        Network<float> net;
        net.add<Dropout>(0.5f, /*seed=*/7);

        net.eval();
        Matrix<float> X(5, 4, 1.0f);
        Matrix<float> Y1 = net.forward(X);
        Matrix<float> Y2 = net.forward(X);
        check(matClose(Y1, Y2) && matClose(Y1, X),
              "eval mode: forward is deterministic and equals input");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
