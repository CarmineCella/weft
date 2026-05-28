// test_sigmoid.cpp
//
// Tests for the Sigmoid activation layer.
//
#include "Sigmoid.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
using weft::Sigmoid;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

int main() {
    std::cout << "weft :: Sigmoid tests\n";

    // ---- 1. Known values ----
    //   sigma(0) = 0.5;  sigma(large) -> 1;  sigma(-large) -> 0
    {
        Sigmoid<float> s;
        Matrix<float> Z{{0.0f, 10.0f, -10.0f}};
        Matrix<float> Y = s.forward(Z);
        check(close(Y(0, 0), 0.5f),            "sigma(0) = 0.5");
        check(Y(0, 1) > 0.9999f,               "sigma(10) ~ 1");
        check(Y(0, 2) < 0.0001f,               "sigma(-10) ~ 0");
    }

    // ---- 2. Output always in (0, 1) ----
    {
        Sigmoid<float> s;
        Matrix<float> Z{{-3.0f, -1.0f, 0.5f}, {2.0f, 4.0f, -6.0f}};
        Matrix<float> Y = s.forward(Z);
        bool ok = true;
        for (std::size_t i = 0; i < Y.rows(); ++i)
            for (std::size_t j = 0; j < Y.cols(); ++j)
                if (Y(i, j) <= 0.0f || Y(i, j) >= 1.0f) ok = false;
        check(ok, "output bounded in (0, 1)");
    }

    // ---- 3. Backward: derivative is Y(1-Y) when dY = 1 ----
    {
        Sigmoid<float> s;
        Matrix<float> Z{{0.0f, 1.0f}};
        Matrix<float> Y = s.forward(Z);
        Matrix<float> dY{{1.0f, 1.0f}};
        Matrix<float> dZ = s.backward(dY);
        // At Z=0: Y=0.5, deriv = 0.25
        check(close(dZ(0, 0), 0.25f), "backward: sigma'(0) = 0.25");
        // At Z=1: Y=sigma(1)~0.7311, deriv = Y(1-Y) ~ 0.1966
        float y1 = 1.0f / (1.0f + std::exp(-1.0f));
        check(close(dZ(0, 1), y1 * (1 - y1)), "backward: sigma'(1) = Y(1-Y)");
    }

    // ---- 4. Numerical gradient check ----
    //   Define a scalar L = sum(sigma(Z)); then dL/dZ = sigma'(Z).
    {
        Sigmoid<float> s;
        Matrix<float> Z{{0.3f, -0.7f, 1.5f}, {-2.0f, 0.0f, 0.9f}};
        Matrix<float> Y = s.forward(Z);
        // dL/dY = 1 everywhere (L = sum of outputs)
        Matrix<float> dY(Z.rows(), Z.cols());
        for (std::size_t i = 0; i < Z.rows(); ++i)
            for (std::size_t j = 0; j < Z.cols(); ++j) dY(i, j) = 1.0f;
        Matrix<float> analytic = s.backward(dY);

        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < Z.rows() && ok; ++i)
            for (std::size_t j = 0; j < Z.cols() && ok; ++j) {
                auto sig = [](float x){ return 1.0f / (1.0f + std::exp(-x)); };
                // L = sum(sigma), so perturbing Z(i,j) only changes that term.
                float Lp = sig(Z(i, j) + eps);
                float Lm = sig(Z(i, j) - eps);
                float numeric = (Lp - Lm) / (2 * eps);
                if (!close(numeric, analytic(i, j), 1e-2f)) ok = false;
            }
        check(ok, "numerical gradient matches Y(1-Y)");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
