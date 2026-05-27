// test_dense.cpp
//
// Tests for the Dense layer:
//   1. forward pass matches a by-hand calculation
//   2. backward pass matches a by-hand calculation
//   3. analytical gradients match numerical gradients (THE essential test)
//   4. SGD actually reduces a quadratic loss (end-to-end sanity)
//
#include "Dense.h"
#include "Matrix.h"

#include <cmath>
#include <iostream>
#include <string>

using weft::Matrix;
using weft::Dense;

// ---- minimal scaffolding -------------------------------------------------
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
static float sumAll(const Matrix<float>& M) {
    float s = 0;
    for (std::size_t i = 0; i < M.rows(); ++i)
        for (std::size_t j = 0; j < M.cols(); ++j)
            s += M(i, j);
    return s;
}

int main() {
    std::cout << "weft :: Dense tests\n";

    // ---- 1. Forward pass, known case ----
    {
        Dense<float> layer(2, 2);
        layer.W() = Matrix<float>{{1, 2}, {3, 4}};
        layer.b() = Matrix<float>{{0}, {0}};

        Matrix<float> X{{1}, {1}};                       // (2 in x 1 example)
        Matrix<float> Z = layer.forward(X);              // = W*X = [[3],[7]]
        check(matClose(Z, Matrix<float>{{3}, {7}}),
              "forward:  Z = W*X + b");
    }

    // ---- 2. Backward pass, same known case ----
    //   dW = dZ * X^T              = [[1],[1]] * [[1,1]] = [[1,1],[1,1]]
    //   db = dZ.sumColumns()       = [[1],[1]]
    //   dX = W^T * dZ              = [[1,3],[2,4]] * [[1],[1]] = [[4],[6]]
    {
        Dense<float> layer(2, 2);
        layer.W() = Matrix<float>{{1, 2}, {3, 4}};
        layer.b() = Matrix<float>{{0}, {0}};

        Matrix<float> X{{1}, {1}};
        layer.forward(X);

        Matrix<float> dZ{{1}, {1}};
        Matrix<float> dX = layer.backward(dZ);

        check(matClose(layer.dW(), Matrix<float>{{1, 1}, {1, 1}}), "backward: dW");
        check(matClose(layer.db(), Matrix<float>{{1}, {1}}),       "backward: db");
        check(matClose(dX,         Matrix<float>{{4}, {6}}),       "backward: dX");
    }

    // ---- 3. Numerical gradient check ----
    // Pick L = sum of all elements of Z, so dL/dZ is a matrix of ones.
    // For each parameter p, compare backward()'s gradient against the
    // numerical estimate (L(p+eps) - L(p-eps)) / (2*eps).
    {
        const std::size_t in = 3, out = 4, batch = 5;
        Dense<float> layer(in, out);

        Matrix<float> X(in, batch);
        X.randomizeUniform(-1.f, 1.f, /*seed=*/42);

        // Analytical: one forward, one backward with dL/dZ = ones.
        Matrix<float> Z  = layer.forward(X);
        Matrix<float> dZ(Z.rows(), Z.cols(), 1.f);
        layer.backward(dZ);

        const float eps = 1e-3f;
        bool dW_ok = true, db_ok = true;

        for (std::size_t i = 0; i < out && dW_ok; ++i) {
            for (std::size_t j = 0; j < in && dW_ok; ++j) {
                float orig = layer.W()(i, j);

                layer.W()(i, j) = orig + eps;
                float L_plus  = sumAll(layer.forward(X));
                layer.W()(i, j) = orig - eps;
                float L_minus = sumAll(layer.forward(X));
                layer.W()(i, j) = orig;

                float numerical  = (L_plus - L_minus) / (2 * eps);
                float analytical = layer.dW()(i, j);
                if (!close(numerical, analytical, 1e-2f)) dW_ok = false;
            }
        }
        check(dW_ok, "numerical gradient check: dW matches backward()");

        for (std::size_t i = 0; i < out && db_ok; ++i) {
            float orig = layer.b()(i, 0);
            layer.b()(i, 0) = orig + eps;
            float L_plus  = sumAll(layer.forward(X));
            layer.b()(i, 0) = orig - eps;
            float L_minus = sumAll(layer.forward(X));
            layer.b()(i, 0) = orig;

            float numerical  = (L_plus - L_minus) / (2 * eps);
            float analytical = layer.db()(i, 0);
            if (!close(numerical, analytical, 1e-2f)) db_ok = false;
        }
        check(db_ok, "numerical gradient check: db matches backward()");
    }

    // ---- 4. End-to-end: SGD on a quadratic loss should reduce it ----
    //   L(W) = 0.5 * sum( (W*X + b - target)^2 )
    //   dL/dZ = Z - target
    // Linear fit -> loss should drop sharply within a few dozen steps.
    {
        Dense<float> layer(3, 2);
        Matrix<float> X{{1, 0, -1}, {0, 1, 1}, {1, -1, 0}};
        Matrix<float> target{{1, 0, 0}, {0, 1, 0}};

        auto loss_and_grad = [&](Matrix<float>& dZ_out) {
            Matrix<float> Z = layer.forward(X);
            Matrix<float> diff = Z - target;
            float L = 0;
            for (std::size_t i = 0; i < diff.rows(); ++i)
                for (std::size_t j = 0; j < diff.cols(); ++j)
                    L += 0.5f * diff(i, j) * diff(i, j);
            dZ_out = diff;
            return L;
        };

        Matrix<float> dZ;
        float L0 = loss_and_grad(dZ);
        for (int step = 0; step < 50; ++step) {
            loss_and_grad(dZ);
            layer.backward(dZ);
            layer.update(0.05f);
        }
        float L_final = loss_and_grad(dZ);
        check(L_final < L0 * 0.1f, "SGD reduces a quadratic loss by >10x in 50 steps");
    }

    std::cout << "\n" << (g_run - g_failed) << " / " << g_run
              << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
