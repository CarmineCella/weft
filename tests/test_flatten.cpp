// tests/test_flatten.cpp
//
// Tests for the flatten / unflatten free functions that bridge
// Tensor4D and Matrix at the conv->dense boundary.  Both functions
// must agree on the row-ordering convention (c -> h -> w within a
// column, examples-as-columns across), otherwise the gradient flow
// through a conv -> flatten -> dense -> flatten -> conv path is
// silently wrong.

#include "../src/Flatten.h"

#include <iostream>
#include <stdexcept>

using namespace weft;

static int passed = 0, total = 0;

static void check(bool cond, const char* name) {
    ++total;
    if (cond) { ++passed; }
    else      { std::cout << "FAIL: " << name << "\n"; }
}

int main() {
    // ----------------------------------------------------------------
    // 1.  Shape: Tensor4D(N, C, H, W) -> Matrix(C*H*W, N).
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(3, 4, 5, 6);
        X.fill(0.0f);
        Matrix<float> M = flatten(X);
        check(M.rows() == 4 * 5 * 6 && M.cols() == 3,
              "flatten: (N,C,H,W) -> (C*H*W, N)");
    }

    // ----------------------------------------------------------------
    // 2.  Column ordering within an example: c -> h -> w.
    //     Place distinctive values at (0,0,0,0), (0,0,0,1), (0,0,1,0),
    //     (0,1,0,0) and verify they land at the expected row offsets.
    // ----------------------------------------------------------------
    {
        const std::size_t C = 2, H = 3, W = 4;
        Tensor4D<float> X(1, C, H, W);
        X.fill(0.0f);
        X(0, 0, 0, 0) = 10.0f;   // expected row 0
        X(0, 0, 0, 1) = 20.0f;   // expected row 1               (w++ -> +1)
        X(0, 0, 1, 0) = 30.0f;   // expected row W               (h++ -> +W)
        X(0, 1, 0, 0) = 40.0f;   // expected row H*W             (c++ -> +H*W)

        Matrix<float> M = flatten(X);
        check(M(0,         0) == 10.0f, "flatten ordering: (0,0,0,0) -> row 0");
        check(M(1,         0) == 20.0f, "flatten ordering: (0,0,0,1) -> row 1");
        check(M(W,         0) == 30.0f, "flatten ordering: (0,0,1,0) -> row W");
        check(M(H * W,     0) == 40.0f, "flatten ordering: (0,1,0,0) -> row H*W");
    }

    // ----------------------------------------------------------------
    // 3.  One example per column.  With N>1, each example's flattened
    //     features are in a distinct column.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(2, 1, 2, 2);
        X(0, 0, 0, 0) = 1.0f; X(0, 0, 0, 1) = 2.0f;
        X(0, 0, 1, 0) = 3.0f; X(0, 0, 1, 1) = 4.0f;
        X(1, 0, 0, 0) = 5.0f; X(1, 0, 0, 1) = 6.0f;
        X(1, 0, 1, 0) = 7.0f; X(1, 0, 1, 1) = 8.0f;

        Matrix<float> M = flatten(X);   // (4, 2)
        check(M(0, 0) == 1.0f && M(1, 0) == 2.0f
           && M(2, 0) == 3.0f && M(3, 0) == 4.0f,
              "flatten: example 0 in column 0");
        check(M(0, 1) == 5.0f && M(1, 1) == 6.0f
           && M(2, 1) == 7.0f && M(3, 1) == 8.0f,
              "flatten: example 1 in column 1");
    }

    // ----------------------------------------------------------------
    // 4.  Round-trip:  unflatten(flatten(X)) == X.  This is the property
    //     the backward pass relies on -- if the orderings of flatten and
    //     unflatten disagreed, the gradient routing through this bridge
    //     would silently scramble.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(2, 3, 4, 5);
        X.randomizeNormal(0.0f, 1.0f, 7u);

        Matrix<float>    M  = flatten(X);
        Tensor4D<float>  Y  = unflatten(M, X.N(), X.C(), X.H(), X.W());

        bool same = true;
        for (std::size_t k = 0; k < X.size() && same; ++k)
            if (X.data()[k] != Y.data()[k]) same = false;
        check(same, "round-trip: unflatten(flatten(X)) == X");
    }

    // ----------------------------------------------------------------
    // 5.  unflatten validates its dimensions.
    // ----------------------------------------------------------------
    {
        Matrix<float> M(12, 2);   // 12 = 1*3*4
        bool threw_rows = false;
        try {
            (void)unflatten(M, /*N=*/2, /*C=*/1, /*H=*/3, /*W=*/5);   // C*H*W=15
        } catch (const std::invalid_argument&) { threw_rows = true; }
        check(threw_rows, "unflatten: rejects row-count mismatch");

        bool threw_cols = false;
        try {
            (void)unflatten(M, /*N=*/3, /*C=*/1, /*H=*/3, /*W=*/4);   // N != cols
        } catch (const std::invalid_argument&) { threw_cols = true; }
        check(threw_cols, "unflatten: rejects col-count mismatch");
    }

    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
