// tests/test_relu4d.cpp
//
// ReLU4D unit tests.  Same checks as test_relu does for Matrix-ReLU,
// just on 4-D tensors.

#include "../src/ReLU4D.h"
#include "../src/SGD.h"

#include <iostream>
#include <cmath>

using namespace weft;

static int passed = 0, total = 0;

static void check(bool cond, const char* name) {
    ++total;
    if (cond) { ++passed; }
    else      { std::cout << "FAIL: " << name << "\n"; }
}

int main() {
    // ----------------------------------------------------------------
    // 1.  Forward zeroes negatives, passes positives.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 1, 2, 2);
        X(0,0,0,0) = -1.0f;
        X(0,0,0,1) =  0.0f;
        X(0,0,1,0) =  2.5f;
        X(0,0,1,1) = -3.0f;
        ReLU4D<float> r;
        Tensor4D<float> Y = r.forward(X);
        check(Y(0,0,0,0) == 0.0f
           && Y(0,0,0,1) == 0.0f
           && Y(0,0,1,0) == 2.5f
           && Y(0,0,1,1) == 0.0f,
              "ReLU4D forward: max(0, x) elementwise");
    }

    // ----------------------------------------------------------------
    // 2.  Forward preserves shape.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(3, 4, 5, 6);
        X.randomizeNormal(0.0f, 1.0f, 7u);
        ReLU4D<float> r;
        Tensor4D<float> Y = r.forward(X);
        check(Y.N() == 3 && Y.C() == 4 && Y.H() == 5 && Y.W() == 6,
              "ReLU4D forward: shape preserved");
    }

    // ----------------------------------------------------------------
    // 3.  Backward routes gradient through positives only.
    //     dx = dy * (x > 0)
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 1, 2, 2);
        X(0,0,0,0) = -1.0f; X(0,0,0,1) =  0.5f;
        X(0,0,1,0) =  2.0f; X(0,0,1,1) = -0.1f;

        ReLU4D<float> r;
        (void)r.forward(X);

        Tensor4D<float> dY(1, 1, 2, 2);
        dY(0,0,0,0) = 10.0f; dY(0,0,0,1) = 20.0f;
        dY(0,0,1,0) = 30.0f; dY(0,0,1,1) = 40.0f;

        Tensor4D<float> dX = r.backward(dY);

        // x < 0 -> dx = 0;   x > 0 -> dx = dy
        check(dX(0,0,0,0) == 0.0f
           && dX(0,0,0,1) == 20.0f
           && dX(0,0,1,0) == 30.0f
           && dX(0,0,1,1) == 0.0f,
              "ReLU4D backward: gradient masked by (x > 0)");
    }

    // ----------------------------------------------------------------
    // 4.  All-negative input zeros everything; all-positive passes through.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> Xn(1, 2, 2, 2, -1.5f);
        ReLU4D<float> r1;
        Tensor4D<float> Yn = r1.forward(Xn);
        bool all_zero = true;
        for (std::size_t k = 0; k < Yn.size(); ++k)
            if (Yn.data()[k] != 0.0f) { all_zero = false; break; }
        check(all_zero, "ReLU4D: all-negative input -> all zero");

        Tensor4D<float> Xp(1, 2, 2, 2,  3.0f);
        ReLU4D<float> r2;
        Tensor4D<float> Yp = r2.forward(Xp);
        bool all_three = true;
        for (std::size_t k = 0; k < Yp.size(); ++k)
            if (Yp.data()[k] != 3.0f) { all_three = false; break; }
        check(all_three, "ReLU4D: all-positive input passes unchanged");
    }

    // ----------------------------------------------------------------
    // 5.  update() is a no-op for activation layers (just verify it
    //     compiles and runs without throwing).
    // ----------------------------------------------------------------
    {
        ReLU4D<float> r;
        SGD<float> opt(0.01f);
        r.update(opt);   // must not throw
        check(true, "ReLU4D::update is a no-op");
    }

    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
