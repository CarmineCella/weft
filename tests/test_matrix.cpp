// test_matrix.cpp
//
// A tiny, dependency-free test harness for weft::Matrix.
// It doubles as documentation: each block shows how an operation is used
// and what it should produce.
//
// Build & run:
//     g++ -std=c++17 test_matrix.cpp -o test_matrix && ./test_matrix
//
#include "Matrix.h"

#include <cmath>
#include <string>
#include <iostream>

using weft::Matrix;

// ---- minimal test scaffolding (no gtest, no dependencies) ----------------
static int g_run = 0;
static int g_failed = 0;

static void check(bool cond, const std::string& name) {
    ++g_run;
    if (cond) {
        std::cout << "  [ ok ] " << name << '\n';
    } else {
        ++g_failed;
        std::cout << "  [FAIL] " << name << '\n';
    }
}

// Element-wise float comparison with a small tolerance.
static bool close(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// Are two matrices equal (same shape, all entries close)?
static bool matClose(const Matrix<float>& A, const Matrix<float>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            if (!close(A(i, j), B(i, j))) return false;
    return true;
}

int main() {
    std::cout << "weft :: Matrix tests\n";

    // --- construction & element access ---
    {
        Matrix<float> A(2, 3);            // zero-initialised
        check(A.rows() == 2 && A.cols() == 3, "shape of (2x3)");
        check(close(A(1, 2), 0.f), "zero-initialised");
        A(1, 2) = 7.f;
        check(close(A(1, 2), 7.f), "element write/read");
    }

    // --- initializer-list construction ---
    {
        Matrix<float> A{{1, 2}, {3, 4}};
        check(A.rows() == 2 && A.cols() == 2, "initializer-list shape");
        check(close(A(0, 1), 2.f) && close(A(1, 0), 3.f), "initializer-list values");
    }

    // --- matrix multiplication: known result ---
    {
        Matrix<float> A{{1, 2}, {3, 4}};
        Matrix<float> B{{5, 6}, {7, 8}};
        Matrix<float> expected{{19, 22}, {43, 50}};
        check(matClose(A * B, expected), "matmul (2x2)*(2x2)");
    }

    // --- transpose ---
    {
        Matrix<float> A{{1, 2, 3}, {4, 5, 6}};
        Matrix<float> expected{{1, 4}, {2, 5}, {3, 6}};
        check(matClose(A.transpose(), expected), "transpose (2x3)->(3x2)");
    }

    // --- Hadamard (element-wise) product ---
    {
        Matrix<float> A{{1, 2}, {3, 4}};
        Matrix<float> B{{2, 2}, {2, 2}};
        Matrix<float> expected{{2, 4}, {6, 8}};
        check(matClose(weft::hadamard(A, B), expected), "hadamard");
    }

    // --- scalar multiply (both orders) ---
    {
        Matrix<float> A{{1, 2}, {3, 4}};
        Matrix<float> expected{{2, 4}, {6, 8}};
        check(matClose(A * 2.f, expected), "matrix * scalar");
        check(matClose(2.f * A, expected), "scalar * matrix");
    }

    // --- element-wise add (same shape) ---
    {
        Matrix<float> A{{1, 1}, {2, 2}};
        Matrix<float> B{{10, 10}, {20, 20}};
        Matrix<float> expected{{11, 11}, {22, 22}};
        check(matClose(A + B, expected), "element-wise add");
    }

    // --- column broadcast add (bias-style) ---
    {
        Matrix<float> M{{1, 1, 1}, {2, 2, 2}};   // (2 x 3): 3 examples
        Matrix<float> b{{10}, {20}};             // (2 x 1): one bias per row
        Matrix<float> expected{{11, 11, 11}, {22, 22, 22}};
        check(matClose(M + b, expected), "broadcast add of (rx1) column");
    }

    // --- sumColumns: sum over the batch ---
    {
        Matrix<float> M{{1, 2, 3}, {4, 5, 6}};
        Matrix<float> expected{{6}, {15}};
        check(matClose(M.sumColumns(), expected), "sumColumns -> (rx1)");
    }

    // --- the payoff: a linear layer forward pass reads like math ---
    {
        Matrix<float> W{{1, 1}, {1, -1}};        // (2 out x 2 in)
        Matrix<float> X{{1, 2}, {3, 4}};         // (2 in x 2 examples)
        Matrix<float> b{{1}, {0}};               // (2 x 1) bias
        Matrix<float> Z = W * X + b;             // forward pass
        Matrix<float> expected{{5, 7}, {-2, -2}};
        check(matClose(Z, expected), "forward pass: Z = W*X + b");
    }

    // --- dimension mismatch should throw ---
    {
        Matrix<float> A(2, 2);
        Matrix<float> C(3, 3);
        bool threw = false;
        try { Matrix<float> bad = A * C; }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "matmul dimension mismatch throws");
    }

    // --- summary ---
    std::cout << "\n" << (g_run - g_failed) << " / " << g_run << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
