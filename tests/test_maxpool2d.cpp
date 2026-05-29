// tests/test_maxpool2d.cpp
//
// MaxPool2D unit tests.  In addition to the shape and hand-computed
// pool checks, the critical bit is a NUMERICAL GRADIENT CHECK that
// verifies backward routes dY to exactly the cached argmax positions.
//
// As with Conv2D's gradient check, this runs in double precision so
// central differences agree with the analytical backward to ~1e-6;
// the dominating risk in MaxPool2D is an off-by-one in either the
// pool-window scan or the cached index encoding, both of which would
// fail the gradient check by orders of magnitude.

#include "../src/MaxPool2D.h"
#include "../src/Optimizer.h"

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace weft;

static int passed = 0, total = 0;

static void check(bool cond, const char* name) {
    ++total;
    if (cond) { ++passed; }
    else      { std::cout << "FAIL: " << name << "\n"; }
}

template <typename T>
static T dot(const Tensor4D<T>& A, const Tensor4D<T>& B) {
    T s = T(0);
    const T* a = A.data();
    const T* b = B.data();
    for (std::size_t k = 0; k < A.size(); ++k) s += a[k] * b[k];
    return s;
}

int main() {
    // ----------------------------------------------------------------
    // 1.  Output shape.  (H - pool) / stride + 1.
    // ----------------------------------------------------------------
    {
        MaxPool2D<float> pool(2);                  // default stride = pool
        Tensor4D<float> X(2, 3, 8, 8); X.fill(0.0f);
        Tensor4D<float> Y = pool.forward(X);
        check(Y.N() == 2 && Y.C() == 3 && Y.H() == 4 && Y.W() == 4,
              "MaxPool2D 2x2 s=2: halves spatial dims");
    }
    {
        MaxPool2D<float> pool(3, 1);               // overlapping
        Tensor4D<float> X(1, 1, 6, 6); X.fill(0.0f);
        Tensor4D<float> Y = pool.forward(X);
        check(Y.H() == 4 && Y.W() == 4,
              "MaxPool2D 3x3 s=1: (H - k)/s + 1 = 4");
    }

    // ----------------------------------------------------------------
    // 2.  Hand-computed forward.  A 4x4 input with known maxima.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 1, 4, 4);
        //   1   2   3   4
        //   5   6   7   8
        //   9  10  11  12
        //  13  14  15  16
        float val = 1.0f;
        for (std::size_t h = 0; h < 4; ++h)
            for (std::size_t w = 0; w < 4; ++w)
                X(0, 0, h, w) = val++;

        MaxPool2D<float> pool(2);   // 2x2, stride 2
        Tensor4D<float> Y = pool.forward(X);
        // Expected windows:
        //   {1,2,5,6}=6   {3,4,7,8}=8
        //   {9,10,13,14}=14   {11,12,15,16}=16
        check(Y(0,0,0,0) ==  6.0f
           && Y(0,0,0,1) ==  8.0f
           && Y(0,0,1,0) == 14.0f
           && Y(0,0,1,1) == 16.0f,
              "MaxPool2D forward: picks window maxima correctly");
    }

    // ----------------------------------------------------------------
    // 3.  Backward routes gradient to argmax positions ONLY.
    //     For the same 4x4 input as above, dY=[[1,2],[3,4]] should give
    //     dX with non-zero only at the corner positions (where 6, 8,
    //     14, 16 lived in the input).
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 1, 4, 4);
        float val = 1.0f;
        for (std::size_t h = 0; h < 4; ++h)
            for (std::size_t w = 0; w < 4; ++w)
                X(0, 0, h, w) = val++;

        MaxPool2D<float> pool(2);
        (void)pool.forward(X);

        Tensor4D<float> dY(1, 1, 2, 2);
        dY(0,0,0,0) = 1.0f; dY(0,0,0,1) = 2.0f;
        dY(0,0,1,0) = 3.0f; dY(0,0,1,1) = 4.0f;

        Tensor4D<float> dX = pool.backward(dY);

        // Sum of dX should equal sum of dY (gradient is just routed, not duplicated or lost).
        float sum = 0.0f;
        for (std::size_t k = 0; k < dX.size(); ++k) sum += dX.data()[k];
        check(std::fabs(sum - 10.0f) < 1e-6f,
              "MaxPool2D backward: total gradient conserved");

        // 6 was at (1,1); 8 was at (1,3); 14 was at (3,1); 16 was at (3,3).
        check(dX(0,0,1,1) == 1.0f
           && dX(0,0,1,3) == 2.0f
           && dX(0,0,3,1) == 3.0f
           && dX(0,0,3,3) == 4.0f,
              "MaxPool2D backward: gradient lands at cached argmax positions");

        // And the non-argmax positions are zero.
        bool rest_zero = true;
        for (std::size_t h = 0; h < 4 && rest_zero; ++h)
            for (std::size_t w = 0; w < 4 && rest_zero; ++w) {
                const bool is_argmax = (h == 1 && (w == 1 || w == 3))
                                    || (h == 3 && (w == 1 || w == 3));
                if (!is_argmax && dX(0,0,h,w) != 0.0f) rest_zero = false;
            }
        check(rest_zero, "MaxPool2D backward: non-argmax positions receive zero");
    }

    // ----------------------------------------------------------------
    // 4.  NUMERICAL GRADIENT CHECK in double precision.
    //
    //     Same setup as Conv2D's grad check: a scalar loss L = <dY, Y>,
    //     so dL/dX[i] = dX[i].  Verify with central differences.
    //
    //     The max operation is non-differentiable where two values tie,
    //     so we use random Gaussian input -- ties have measure zero at
    //     double precision.
    // ----------------------------------------------------------------
    {
        const std::size_t N = 2, C = 2, H = 6, W = 6;
        Tensor4D<double> X(N, C, H, W);
        X.randomizeNormal(0.0, 1.0, /*seed=*/42u);

        MaxPool2D<double> pool(2);   // 2x2, stride 2 -> output (N, C, 3, 3)
        Tensor4D<double> Y = pool.forward(X);

        Tensor4D<double> dY(Y.N(), Y.C(), Y.H(), Y.W());
        dY.randomizeNormal(0.0, 1.0, /*seed=*/43u);

        Tensor4D<double> dX_an = pool.backward(dY);

        const double eps     = 1e-5;
        const double tol_rel = 1e-6;
        const double tol_abs = 1e-8;
        auto ok = [&](double num, double an) {
            const double diff = std::fabs(num - an);
            if (diff <= tol_abs) return true;
            const double denom = std::max(std::fabs(num), std::fabs(an));
            return diff / denom <= tol_rel;
        };

        bool grad_ok = true;
        for (std::size_t n = 0; n < N && grad_ok; ++n)
        for (std::size_t c = 0; c < C && grad_ok; ++c)
        for (std::size_t h = 0; h < H && grad_ok; ++h)
        for (std::size_t w = 0; w < W && grad_ok; ++w) {
            const double saved = X(n, c, h, w);
            X(n, c, h, w) = saved + eps;
            Tensor4D<double> Yp = pool.forward(X);
            X(n, c, h, w) = saved - eps;
            Tensor4D<double> Ym = pool.forward(X);
            X(n, c, h, w) = saved;
            const double num = (dot(dY, Yp) - dot(dY, Ym)) / (2.0 * eps);
            const double an  = dX_an(n, c, h, w);
            if (!ok(num, an)) {
                std::cout << "  dX mismatch at (" << n << "," << c << ","
                          << h << "," << w << "): num=" << num
                          << " analytical=" << an << "\n";
                grad_ok = false;
            }
        }
        check(grad_ok, "MaxPool2D: backward dX matches numerical (central diff)");
    }

    // ----------------------------------------------------------------
    // 5.  Stride 1 (overlapping windows).  A single input pixel can win
    //     in multiple output windows; its gradient must accumulate.
    // ----------------------------------------------------------------
    {
        Tensor4D<double> X(1, 1, 4, 4);
        X.randomizeNormal(0.0, 1.0, 99u);

        MaxPool2D<double> pool(2, 1);   // 2x2, stride 1 -> output 3x3
        Tensor4D<double> Y = pool.forward(X);
        Tensor4D<double> dY(Y.N(), Y.C(), Y.H(), Y.W());
        dY.randomizeNormal(0.0, 1.0, 100u);
        Tensor4D<double> dX_an = pool.backward(dY);

        const double eps = 1e-5;
        const double tol_rel = 1e-6;
        const double tol_abs = 1e-8;
        auto ok = [&](double num, double an) {
            const double diff = std::fabs(num - an);
            if (diff <= tol_abs) return true;
            const double denom = std::max(std::fabs(num), std::fabs(an));
            return diff / denom <= tol_rel;
        };

        bool grad_ok = true;
        for (std::size_t h = 0; h < 4 && grad_ok; ++h)
        for (std::size_t w = 0; w < 4 && grad_ok; ++w) {
            const double saved = X(0, 0, h, w);
            X(0, 0, h, w) = saved + eps;
            Tensor4D<double> Yp = pool.forward(X);
            X(0, 0, h, w) = saved - eps;
            Tensor4D<double> Ym = pool.forward(X);
            X(0, 0, h, w) = saved;
            const double num = (dot(dY, Yp) - dot(dY, Ym)) / (2.0 * eps);
            const double an  = dX_an(0, 0, h, w);
            if (!ok(num, an)) grad_ok = false;
        }
        check(grad_ok, "MaxPool2D: gradient correct at stride=1 (overlapping windows)");
    }

    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
