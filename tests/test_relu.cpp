// test_relu.cpp
//
// Tests for the ReLU activation layer.
//
#include "ReLU.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
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
static float sumSquares(const Matrix<float>& M) {
    float s = 0;
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            s += M(i, j) * M(i, j);
    return s;
}

int main() {
    std::cout << "weft :: ReLU tests\n";

    // ---- Forward: max(0, z) ----
    {
        ReLU<float> relu;
        Matrix<float> Z{{-1, 0}, {1, 2}};
        Matrix<float> A = relu.forward(Z);
        check(matClose(A, Matrix<float>{{0, 0}, {1, 2}}),
              "forward:  A = max(0, Z)");
    }

    // ---- Backward: pass-through where Z > 0, zero elsewhere ----
    {
        ReLU<float> relu;
        Matrix<float> Z{{-1, 0.5f}, {1, -2}};
        relu.forward(Z);
        Matrix<float> dA{{10, 20}, {30, 40}};
        Matrix<float> dZ = relu.backward(dA);
        // mask = [[0, 1], [1, 0]]  ->  dZ = [[0, 20], [30, 0]]
        check(matClose(dZ, Matrix<float>{{0, 20}, {30, 0}}),
              "backward: dZ = dA * (Z > 0)");
    }

    // ---- Numerical gradient check ----
    // Use L = 0.5 * sum(A^2) so dL/dA = A.
    // Inputs are chosen well away from zero so the perturbation never
    // straddles the kink at z = 0.
    {
        ReLU<float> relu;
        Matrix<float> Z{{-0.7f,  0.5f, -0.3f,  0.8f},
                        { 0.7f, -0.4f,  0.2f, -0.9f},
                        { 0.6f,  0.3f, -0.7f,  0.4f}};
        Matrix<float> A = relu.forward(Z);
        Matrix<float> dA = A;                       // dL/dA for L = 0.5 sum A^2
        Matrix<float> dZ = relu.backward(dA);

        const float eps = 1e-3f;
        bool ok = true;
        for (std::size_t i = 0; i < Z.rows() && ok; ++i) {
            for (std::size_t j = 0; j < Z.cols() && ok; ++j) {
                float orig = Z(i, j);
                Z(i, j) = orig + eps;
                float L_plus  = 0.5f * sumSquares(relu.forward(Z));
                Z(i, j) = orig - eps;
                float L_minus = 0.5f * sumSquares(relu.forward(Z));
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
