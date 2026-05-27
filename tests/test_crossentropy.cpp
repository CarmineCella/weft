// test_crossentropy.cpp
//
// Tests for CrossEntropy loss.
//
//   1. forward on a hand-checkable case
//   2. forward averages correctly over a batch
//   3. numerical gradient check on dS  (vs CE.backward())
//   4. CE.backward chained through Softmax.backward == (S - T)/N
//      (the famous Softmax+CE shortcut, verified)
//   5. full-chain numerical gradient check: perturb Z, confirm dZ
//
#include "CrossEntropy.h"
#include "Softmax.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
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
    std::cout << "weft :: CrossEntropy tests\n";

    // ---- 1. Forward: single example, hand-checkable ----
    // S = (0.7, 0.2, 0.1)^T, T = one-hot at class 0
    // L = -log(0.7) ~= 0.35667
    {
        CrossEntropy<float> ce;
        Matrix<float> S{{0.7f}, {0.2f}, {0.1f}};
        Matrix<float> T{{1.0f}, {0.0f}, {0.0f}};
        float L = ce.forward(S, T);
        check(close(L, -std::log(0.7f), 1e-5f),
              "forward:  L = -log(S[correct class]) for one-hot target");
    }

    // ---- 2. Forward: batch averages per-example losses ----
    // Example 0: target class 0, S[0]=0.7  ->  -log(0.7)
    // Example 1: target class 1, S[1]=0.8  ->  -log(0.8)
    // Expected mean loss = ( -log(0.7) + -log(0.8) ) / 2
    {
        CrossEntropy<float> ce;
        Matrix<float> S{{0.7f, 0.1f},
                        {0.2f, 0.8f},
                        {0.1f, 0.1f}};
        Matrix<float> T{{1.f, 0.f},
                        {0.f, 1.f},
                        {0.f, 0.f}};
        float L = ce.forward(S, T);
        float expected = 0.5f * (-std::log(0.7f) - std::log(0.8f));
        check(close(L, expected, 1e-5f),
              "forward:  batch loss is the mean of per-example losses");
    }

    // ---- 3. Numerical gradient check on dS ----
    //    perturb each S[i,j] independently, check (L+ - L-)/(2 eps) vs backward()
    {
        CrossEntropy<float> ce;
        Matrix<float> S{{0.5f, 0.2f},
                        {0.3f, 0.5f},
                        {0.2f, 0.3f}};
        Matrix<float> T{{1.f, 0.f},
                        {0.f, 1.f},
                        {0.f, 0.f}};
        ce.forward(S, T);
        Matrix<float> dS = ce.backward();

        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < S.rows() && ok; ++i) {
            for (std::size_t j = 0; j < S.cols() && ok; ++j) {
                float orig = S(i, j);
                S(i, j) = orig + eps;
                float L_plus  = ce.forward(S, T);
                S(i, j) = orig - eps;
                float L_minus = ce.forward(S, T);
                S(i, j) = orig;

                float numerical  = (L_plus - L_minus) / (2 * eps);
                float analytical = dS(i, j);
                if (!close(numerical, analytical, 1e-2f)) ok = false;
            }
        }
        check(ok, "numerical gradient check on dS");
    }

    // ---- 4. The Softmax + CE shortcut:  Softmax.backward(CE.backward()) == (S - T)/N ----
    {
        Softmax<float>     sm;
        CrossEntropy<float> ce;

        // Some logits Z, some one-hot targets T
        Matrix<float> Z{{ 1.0f,  0.3f, -0.5f},
                        { 0.5f, -0.2f,  1.0f},
                        {-0.3f,  0.8f,  0.2f}};
        Matrix<float> T{{1.f, 0.f, 0.f},
                        {0.f, 1.f, 0.f},
                        {0.f, 0.f, 1.f}};

        Matrix<float> S  = sm.forward(Z);
        ce.forward(S, T);
        Matrix<float> dS = ce.backward();
        Matrix<float> dZ_composed = sm.backward(dS);

        // Compute the shortcut form (S - T) / N directly
        const float invN = 1.f / static_cast<float>(S.cols());
        Matrix<float> dZ_direct(S.rows(), S.cols());
        for (std::size_t i = 0; i < S.rows(); ++i)
            for (std::size_t j = 0; j < S.cols(); ++j)
                dZ_direct(i, j) = invN * (S(i, j) - T(i, j));

        check(matClose(dZ_composed, dZ_direct, 1e-5f),
              "Softmax.backward(CE.backward())  ==  (S - T) / N");
    }

    // ---- 5. Full-chain numerical gradient check on Z ----
    //    Perturb each logit Zij, confirm (L+ - L-)/(2 eps) == dZ from chain
    {
        Softmax<float>      sm;
        CrossEntropy<float> ce;

        Matrix<float> Z{{ 1.0f,  0.3f},
                        { 0.5f, -0.2f},
                        {-0.3f,  0.8f}};
        Matrix<float> T{{1.f, 0.f},
                        {0.f, 1.f},
                        {0.f, 0.f}};

        Matrix<float> S  = sm.forward(Z);
        ce.forward(S, T);
        Matrix<float> dZ = sm.backward(ce.backward());

        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < Z.rows() && ok; ++i) {
            for (std::size_t j = 0; j < Z.cols() && ok; ++j) {
                float orig = Z(i, j);
                Z(i, j) = orig + eps;
                float L_plus  = ce.forward(sm.forward(Z), T);
                Z(i, j) = orig - eps;
                float L_minus = ce.forward(sm.forward(Z), T);
                Z(i, j) = orig;

                float numerical  = (L_plus - L_minus) / (2 * eps);
                float analytical = dZ(i, j);
                if (!close(numerical, analytical, 1e-2f)) ok = false;
            }
        }
        check(ok, "full chain:  numerical dZ matches Softmax->CE backward");
    }

    std::cout << "\n" << (g_run - g_failed) << " / " << g_run
              << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
