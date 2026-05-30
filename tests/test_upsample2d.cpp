// tests/test_upsample2d.cpp
//
// Upsample2D unit tests.  The forward is trivial -- replicate each input
// pixel into a factor x factor block.  The interesting check is the
// BACKWARD: every output position contributes its dY back to the SAME
// input pixel, so the input gradient at (h, w) is the sum of dY over
// the factor x factor output block.
//
// As elsewhere in weft, the gradient check runs in double precision so
// the analytical and central-difference gradients agree to ~1e-6.

#include "../src/Upsample2D.h"
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
    // 1.  Output shape: H_out = H * factor, W_out = W * factor.
    // ----------------------------------------------------------------
    {
        Upsample2D<float> up(2);
        Tensor4D<float> X(2, 3, 4, 5); X.fill(0.0f);
        Tensor4D<float> Y = up.forward(X);
        check(Y.N() == 2 && Y.C() == 3 && Y.H() == 8 && Y.W() == 10,
              "Upsample2D factor=2: doubles spatial dims, keeps N,C");
    }
    {
        Upsample2D<float> up(3);
        Tensor4D<float> X(1, 2, 2, 2); X.fill(0.0f);
        Tensor4D<float> Y = up.forward(X);
        check(Y.H() == 6 && Y.W() == 6,
              "Upsample2D factor=3: H/W tripled");
    }
    {
        Upsample2D<float> up(1);     // identity
        Tensor4D<float> X(1, 1, 3, 3); X.fill(7.0f);
        Tensor4D<float> Y = up.forward(X);
        check(Y.H() == 3 && Y.W() == 3 && Y(0,0,0,0) == 7.0f
                                       && Y(0,0,2,2) == 7.0f,
              "Upsample2D factor=1: identity");
    }

    // ----------------------------------------------------------------
    // 2.  Hand-computed forward.  A 2x2 input with distinct values
    //     should produce a 4x4 output of 2x2 blocks, each block holding
    //     the corresponding input value.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 1, 2, 2);
        X(0,0,0,0) = 1.0f;  X(0,0,0,1) = 2.0f;
        X(0,0,1,0) = 3.0f;  X(0,0,1,1) = 4.0f;

        Upsample2D<float> up(2);
        Tensor4D<float> Y = up.forward(X);
        // Expected:
        //   1 1 2 2
        //   1 1 2 2
        //   3 3 4 4
        //   3 3 4 4
        bool ok = true;
        for (std::size_t dy = 0; dy < 2 && ok; ++dy)
            for (std::size_t dx = 0; dx < 2 && ok; ++dx) {
                ok = ok && Y(0,0,    dy,     dx) == 1.0f;
                ok = ok && Y(0,0,    dy, 2 + dx) == 2.0f;
                ok = ok && Y(0,0, 2 +dy,     dx) == 3.0f;
                ok = ok && Y(0,0, 2 +dy, 2 + dx) == 4.0f;
            }
        check(ok, "Upsample2D forward: replicates each input pixel into a 2x2 block");
    }

    // ----------------------------------------------------------------
    // 3.  Backward sums gradient over the factor x factor block.
    //     For X(0,0,0,0)=1, the four output positions Y(0,0,0..1,0..1)
    //     all came from it; with dY=[[1,2,3,4],...], dX(0,0,0,0) should
    //     be the sum of dY in that 2x2 block.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 1, 2, 2); X.fill(0.0f);
        Upsample2D<float> up(2);
        (void)up.forward(X);

        Tensor4D<float> dY(1, 1, 4, 4);
        // Fill dY with 1..16
        float v = 1.0f;
        for (std::size_t h = 0; h < 4; ++h)
            for (std::size_t w = 0; w < 4; ++w)
                dY(0,0,h,w) = v++;

        Tensor4D<float> dX = up.backward(dY);
        // dX(0,0,0,0) = dY(0,0,0,0..1) + dY(0,0,1,0..1) = 1+2+5+6 = 14
        // dX(0,0,0,1) = 3+4+7+8 = 22
        // dX(0,0,1,0) = 9+10+13+14 = 46
        // dX(0,0,1,1) = 11+12+15+16 = 54
        check(dX(0,0,0,0) == 14.0f
           && dX(0,0,0,1) == 22.0f
           && dX(0,0,1,0) == 46.0f
           && dX(0,0,1,1) == 54.0f,
              "Upsample2D backward: dX is sum of dY over factor x factor block");

        // Total gradient is preserved: sum(dX) == sum(dY).
        float sX = 0.0f, sY = 0.0f;
        for (std::size_t k = 0; k < dX.size(); ++k) sX += dX.data()[k];
        for (std::size_t k = 0; k < dY.size(); ++k) sY += dY.data()[k];
        check(std::fabs(sX - sY) < 1e-6f,
              "Upsample2D backward: total gradient conserved");
    }

    // ----------------------------------------------------------------
    // 4.  NUMERICAL GRADIENT CHECK in double precision.
    //
    //     Loss L = <dY, Y>, so dL/dX[i] should equal dX_an[i].
    //     With Upsample2D, dL/dX[i] = sum of dY over the factor x factor
    //     output block that came from input pixel i.  The analytical
    //     backward should reproduce this exactly.
    // ----------------------------------------------------------------
    {
        const std::size_t N = 2, C = 3, H = 4, W = 5, factor = 2;
        Tensor4D<double> X(N, C, H, W);
        X.randomizeNormal(0.0, 1.0, /*seed=*/11u);

        Upsample2D<double> up(factor);
        Tensor4D<double> Y = up.forward(X);

        Tensor4D<double> dY(Y.N(), Y.C(), Y.H(), Y.W());
        dY.randomizeNormal(0.0, 1.0, /*seed=*/12u);

        Tensor4D<double> dX_an = up.backward(dY);

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
            Tensor4D<double> Yp = up.forward(X);
            X(n, c, h, w) = saved - eps;
            Tensor4D<double> Ym = up.forward(X);
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
        check(grad_ok, "Upsample2D: backward dX matches numerical (central diff)");
    }

    // ----------------------------------------------------------------
    // 5.  Composition: MaxPool2D(2) then Upsample2D(2) should give back
    //     a tensor of the original SHAPE.  Values aren't preserved
    //     (pool keeps maxima only) but the shapes must round-trip.
    //     This is the property we lean on in ConvVAE: encoder pools,
    //     decoder upsamples back to the input shape.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 4, 8, 8); X.fill(1.0f);
        // Quick "encoder" shape: MaxPool halves
        // Reuse the test's intent without including MaxPool here -- just
        // verify Upsample2D doubles the input dimensions we'd get after
        // pooling.
        Tensor4D<float> after_pool(1, 4, 4, 4); after_pool.fill(1.0f);
        Upsample2D<float> up(2);
        Tensor4D<float> back = up.forward(after_pool);
        check(back.N() == 1 && back.C() == 4 && back.H() == 8 && back.W() == 8,
              "Upsample2D round-trips MaxPool(2)'s shape change");
    }

    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
