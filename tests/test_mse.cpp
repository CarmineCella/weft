// test_mse.cpp
//
// Tests for the mean squared error loss.
//
#include "MSE.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
using weft::MSE;

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
    std::cout << "weft :: MSE tests\n";

    // ---- 1. Zero loss when prediction == target ----
    {
        MSE<float> mse;
        Matrix<float> P{{1, 2, 3}, {4, 5, 6}};
        Matrix<float> T = P;
        check(close(mse.forward(P, T), 0.0f), "zero loss when P == T");
    }

    // ---- 2. Known value ----
    //   P - T = [[1, 1], [1, 1]]  (all differences = 1)
    //   sum of squares = 4,  N = 2 columns
    //   L = 4 / (2 * 2) = 1.0
    {
        MSE<float> mse;
        Matrix<float> P{{2, 3}, {4, 5}};
        Matrix<float> T{{1, 2}, {3, 4}};
        check(close(mse.forward(P, T), 1.0f), "known value: L = 1.0");
    }

    // ---- 3. Gradient is (1/N)(P - T) ----
    {
        MSE<float> mse;
        Matrix<float> P{{2, 4}, {6, 8}};
        Matrix<float> T{{1, 1}, {1, 1}};
        mse.forward(P, T);
        Matrix<float> g = mse.backward();
        // N = 2.  dL/dP = 0.5 * (P - T)
        //   P - T = [[1, 3], [5, 7]]
        //   grad  = [[0.5, 1.5], [2.5, 3.5]]
        bool ok = close(g(0, 0), 0.5f) && close(g(0, 1), 1.5f) &&
                  close(g(1, 0), 2.5f) && close(g(1, 1), 3.5f);
        check(ok, "gradient = (1/N)(P - T)");
    }

    // ---- 4. Numerical gradient check ----
    //   Perturb each entry of P, compare finite-difference dL/dP to the
    //   analytic gradient from backward().
    {
        MSE<float> mse;
        Matrix<float> P{{0.5f, -1.2f, 2.0f}, {3.1f, 0.0f, -0.7f}};
        Matrix<float> T{{1.0f,  0.5f, 1.5f}, {2.0f, 0.3f,  0.1f}};

        mse.forward(P, T);
        Matrix<float> analytic = mse.backward();

        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < P.rows() && ok; ++i)
            for (std::size_t j = 0; j < P.cols() && ok; ++j) {
                Matrix<float> Pp = P, Pm = P;
                Pp(i, j) += eps;
                Pm(i, j) -= eps;
                MSE<float> a, b;
                float Lp = a.forward(Pp, T);
                float Lm = b.forward(Pm, T);
                float numeric = (Lp - Lm) / (2 * eps);
                if (!close(numeric, analytic(i, j), 1e-2f)) ok = false;
            }
        check(ok, "numerical gradient matches analytic gradient");
    }

    // ---- 5. Shape mismatch throws ----
    {
        MSE<float> mse;
        Matrix<float> P(2, 3);
        Matrix<float> T(2, 4);
        bool threw = false;
        try { mse.forward(P, T); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "shape mismatch throws");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
