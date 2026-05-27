// test_softmax.cpp
//
// Tests for the Softmax activation layer.
//
#include "Softmax.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
using weft::Softmax;

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
static float sumSquares(const Matrix<float>& M) {
    float s = 0;
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            s += M(i, j) * M(i, j);
    return s;
}

int main() {
    std::cout << "weft :: Softmax tests\n";

    // ---- Forward: each column is a valid probability distribution ----
    {
        Softmax<float> sm;
        Matrix<float> Z{{1.0f, 2.0f}, {2.0f, 0.0f}, {3.0f, -1.0f}};
        Matrix<float> S = sm.forward(Z);
        bool sums_ok = true, positive_ok = true;
        for (std::size_t j = 0; j < S.cols(); ++j) {
            float sum = 0;
            for (std::size_t i = 0; i < S.rows(); ++i) {
                sum += S(i, j);
                if (S(i, j) <= 0) positive_ok = false;
            }
            if (!close(sum, 1.0f)) sums_ok = false;
        }
        check(sums_ok,     "forward:  each column sums to 1");
        check(positive_ok, "forward:  all outputs are strictly positive");
    }

    // ---- Numerical stability: softmax(z) == softmax(z + big constant) ----
    {
        Softmax<float> sm1, sm2;
        Matrix<float> Z{{1.0f}, {2.0f}, {3.0f}};
        Matrix<float> Z_shifted = Z.apply([](float x){ return x + 1000.f; });
        Matrix<float> S1 = sm1.forward(Z);
        Matrix<float> S2 = sm2.forward(Z_shifted);
        check(matClose(S1, S2),
              "forward:  softmax(z) == softmax(z + 1000)  (no overflow)");
    }

    // ---- Equal logits give a uniform distribution ----
    {
        Softmax<float> sm;
        Matrix<float> Z{{5.0f}, {5.0f}, {5.0f}};
        Matrix<float> S = sm.forward(Z);
        check(close(S(0,0), 1.f/3.f) &&
              close(S(1,0), 1.f/3.f) &&
              close(S(2,0), 1.f/3.f),
              "forward:  equal logits -> uniform distribution");
    }

    // ---- Numerical gradient check ----
    // L = 0.5 * sum(S^2),  dL/dS = S.  Two examples (columns), 3 classes.
    {
        Softmax<float> sm;
        Matrix<float> Z{{ 1.0f, -0.5f},
                        { 0.5f,  1.0f},
                        {-0.3f,  0.2f}};
        Matrix<float> S  = sm.forward(Z);
        Matrix<float> dS = S;
        Matrix<float> dZ = sm.backward(dS);

        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < Z.rows() && ok; ++i) {
            for (std::size_t j = 0; j < Z.cols() && ok; ++j) {
                float orig = Z(i, j);
                Z(i, j) = orig + eps;
                float L_plus  = 0.5f * sumSquares(sm.forward(Z));
                Z(i, j) = orig - eps;
                float L_minus = 0.5f * sumSquares(sm.forward(Z));
                Z(i, j) = orig;
                float numerical  = (L_plus - L_minus) / (2 * eps);
                float analytical = dZ(i, j);
                if (!close(numerical, analytical, 1e-2f)) ok = false;
            }
        }
        check(ok, "numerical gradient check");
    }

    std::cout << "\n" << (g_run - g_failed) << " / " << g_run
              << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
