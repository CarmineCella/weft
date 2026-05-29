// tests/test_conv2d.cpp
//
// Conv2D unit tests.  Shape checks confirm the output dimensions match
// the (in_size + 2*padding - kernel) / stride + 1 formula across a
// handful of configurations, and the critical bit -- a NUMERICAL
// GRADIENT CHECK -- confirms backward() agrees with finite differences
// for dX, dW, and db.
//
// Any indexing mistake in im2col or col2im will fail the gradient
// check almost certainly, because the analytical gradient and the
// numerical gradient walk completely different code paths.

#include "../src/Conv2D.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace weft;

static int passed = 0, total = 0;

static void check(bool cond, const char* name) {
    ++total;
    if (cond) { ++passed; }
    else      { std::cout << "FAIL: " << name << "\n"; }
}

// Inner product of two same-shape tensors, used to turn "gradient of
// loss w.r.t. parameter p" into a scalar: loss := <dY, Y> by definition,
// so dL/dp = <dY, dY/dp>, which for any parameter p collapses to
// either the analytical dp (from backward) or the finite-difference
// numerical estimate.
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
    // 1.  Construction: weight matrix shape (out_C, in_C*K*K), bias
    //     shape (out_C, 1).
    // ----------------------------------------------------------------
    {
        Conv2D<double> conv(/*in=*/3, /*out=*/8, /*k=*/3,
                           /*stride=*/1, /*pad=*/1, /*seed=*/1u);
        check(conv.W().rows() == 8 && conv.W().cols() == 3 * 3 * 3,
              "Conv2D ctor: W shape = (out_C, in_C*K*K)");
        check(conv.b().rows() == 8 && conv.b().cols() == 1,
              "Conv2D ctor: b shape = (out_C, 1)");
        check(conv.in_channels() == 3 && conv.out_channels() == 8
           && conv.kernel_size() == 3,
              "Conv2D ctor: stored hyperparameters correct");
    }

    // ----------------------------------------------------------------
    // 2.  Output shapes across stride / padding configurations.
    //     H_out = (H + 2*pad - K) / stride + 1
    // ----------------------------------------------------------------
    {
        Conv2D<double> conv(1, 4, 3, 1, 1, 1u);   // pad=1: same-size
        Tensor4D<double> X(2, 1, 28, 28); X.fill(0.5f);
        Tensor4D<double> Y = conv.forward(X);
        check(Y.N() == 2 && Y.C() == 4 && Y.H() == 28 && Y.W() == 28,
              "Conv2D: 3x3 stride=1 pad=1 preserves H,W");
    }
    {
        Conv2D<double> conv(1, 4, 3, 2, 1, 1u);   // stride=2: halves H,W
        Tensor4D<double> X(1, 1, 28, 28); X.fill(0.0);
        Tensor4D<double> Y = conv.forward(X);
        check(Y.H() == 14 && Y.W() == 14,
              "Conv2D: 3x3 stride=2 pad=1 halves H,W");
    }
    {
        Conv2D<double> conv(2, 3, 2, 1, 0, 1u);   // 2x2 no pad: shrinks by 1
        Tensor4D<double> X(1, 2, 5, 5); X.fill(0.0);
        Tensor4D<double> Y = conv.forward(X);
        check(Y.H() == 4 && Y.W() == 4,
              "Conv2D: 2x2 stride=1 pad=0 shrinks H,W by K-1");
    }

    // ----------------------------------------------------------------
    // 3.  Bias-only test.  With a zero kernel and a constant bias,
    //     every output pixel of channel c should equal b(c).
    // ----------------------------------------------------------------
    {
        Conv2D<double> conv(2, 3, 3, 1, 1, 1u);
        for (std::size_t i = 0; i < conv.W().rows(); ++i)
            for (std::size_t j = 0; j < conv.W().cols(); ++j)
                conv.W()(i, j) = 0.0;
        conv.b()(0, 0) = 1.5f;
        conv.b()(1, 0) = -2.0;
        conv.b()(2, 0) = 7.0f;

        Tensor4D<double> X(1, 2, 4, 4); X.fill(0.3f);
        Tensor4D<double> Y = conv.forward(X);

        bool ok = true;
        for (std::size_t h = 0; h < Y.H() && ok; ++h)
            for (std::size_t w = 0; w < Y.W() && ok; ++w) {
                if (std::fabs(Y(0, 0, h, w) - 1.5f) > 1e-5f) ok = false;
                if (std::fabs(Y(0, 1, h, w) - (-2.0)) > 1e-5f) ok = false;
                if (std::fabs(Y(0, 2, h, w) - 7.0f) > 1e-5f) ok = false;
            }
        check(ok, "Conv2D: zero kernel + bias produces broadcast bias");
    }

    // ----------------------------------------------------------------
    // 4.  Forward channel count must match.  Constructing Conv2D for
    //     2 input channels then passing a 3-channel tensor should throw.
    // ----------------------------------------------------------------
    {
        Conv2D<double> conv(2, 4, 3, 1, 1, 1u);
        Tensor4D<double> bad(1, 3, 4, 4);
        bool threw = false;
        try { (void)conv.forward(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "Conv2D::forward: channel mismatch throws");
    }

    // ----------------------------------------------------------------
    // 5.  *** NUMERICAL GRADIENT CHECK ***
    //
    // Define the scalar loss  L = <dY, Y>  for some randomly-chosen
    // upstream gradient dY.  Then
    //     dL/dX[i] = dX[i]      (the value backward returns)
    //     dL/dW[i] = dW[i]
    //     dL/db[i] = db[i]
    //
    // and we can verify each entry separately with central differences:
    //     dL/dp[i]  ~=  ( L(p + eps) - L(p - eps) ) / (2 eps)
    //
    // Central differences are O(eps^2) accurate, so with eps = 1e-3
    // we can expect agreement to a few parts in 10^-4 for float32.
    // Any indexing bug in im2col or col2im is overwhelmingly likely
    // to show up here.
    // ----------------------------------------------------------------
    {
        const std::size_t N = 2, in_C = 2, out_C = 3;
        const std::size_t K = 3, H = 5, W = 5, pad = 1;

        Conv2D<double> conv(in_C, out_C, K, /*stride=*/1, pad, /*seed=*/42u);

        Tensor4D<double> X(N, in_C, H, W);
        X.randomizeNormal(0.0, 1.0, /*seed=*/123u);

        // Compute analytical gradients from one forward+backward pass.
        Tensor4D<double> Y = conv.forward(X);
        Tensor4D<double> dY(Y.N(), Y.C(), Y.H(), Y.W());
        dY.randomizeNormal(0.0, 1.0, /*seed=*/456u);

        Tensor4D<double> dX_an = conv.backward(dY);
        // Snapshot dW, db before we corrupt the layer with more forwards.
        Matrix<double> dW_an = conv.dW();
        Matrix<double> db_an = conv.db();

        const double eps     = 1e-5;
        const double tol_rel = 1e-6;   // 1% relative -- realistic for fp32 central diff
        const double tol_abs = 1e-8;   // absolute floor for tiny-magnitude gradients

        auto ok = [&](double num, double an) {
            const double diff = std::fabs(num - an);
            if (diff <= tol_abs) return true;
            const double denom = std::max(std::fabs(num), std::fabs(an));
            return diff / denom <= tol_rel;
        };

        // ---- check dX entry by entry --------------------------------
        bool dX_ok = true;
        for (std::size_t n = 0; n < N && dX_ok; ++n)
        for (std::size_t c = 0; c < in_C && dX_ok; ++c)
        for (std::size_t h = 0; h < H && dX_ok; ++h)
        for (std::size_t w = 0; w < W && dX_ok; ++w) {
            const double saved = X(n, c, h, w);
            X(n, c, h, w) = saved + eps;
            Tensor4D<double> Yp = conv.forward(X);
            X(n, c, h, w) = saved - eps;
            Tensor4D<double> Ym = conv.forward(X);
            X(n, c, h, w) = saved;
            const double num = (dot(dY, Yp) - dot(dY, Ym)) / (2.0 * eps);
            const double an  = dX_an(n, c, h, w);
            if (!ok(num, an)) {
                std::cout << "  dX mismatch at (" << n << "," << c << ","
                          << h << "," << w << "): num=" << num
                          << " analytical=" << an << "\n";
                dX_ok = false;
            }
        }
        check(dX_ok, "Conv2D: backward dX matches numerical (central diff)");

        // ---- check dW entry by entry --------------------------------
        bool dW_ok = true;
        for (std::size_t i = 0; i < conv.W().rows() && dW_ok; ++i)
        for (std::size_t j = 0; j < conv.W().cols() && dW_ok; ++j) {
            const double saved = conv.W()(i, j);
            conv.W()(i, j) = saved + eps;
            Tensor4D<double> Yp = conv.forward(X);
            conv.W()(i, j) = saved - eps;
            Tensor4D<double> Ym = conv.forward(X);
            conv.W()(i, j) = saved;
            const double num = (dot(dY, Yp) - dot(dY, Ym)) / (2.0 * eps);
            const double an  = dW_an(i, j);
            if (!ok(num, an)) {
                std::cout << "  dW mismatch at (" << i << "," << j
                          << "): num=" << num << " analytical=" << an << "\n";
                dW_ok = false;
            }
        }
        check(dW_ok, "Conv2D: backward dW matches numerical (central diff)");

        // ---- check db entry by entry --------------------------------
        bool db_ok = true;
        for (std::size_t i = 0; i < conv.b().rows() && db_ok; ++i) {
            const double saved = conv.b()(i, 0);
            conv.b()(i, 0) = saved + eps;
            Tensor4D<double> Yp = conv.forward(X);
            conv.b()(i, 0) = saved - eps;
            Tensor4D<double> Ym = conv.forward(X);
            conv.b()(i, 0) = saved;
            const double num = (dot(dY, Yp) - dot(dY, Ym)) / (2.0 * eps);
            const double an  = db_an(i, 0);
            if (!ok(num, an)) {
                std::cout << "  db mismatch at " << i
                          << ": num=" << num << " analytical=" << an << "\n";
                db_ok = false;
            }
        }
        check(db_ok, "Conv2D: backward db matches numerical (central diff)");
    }

    // ----------------------------------------------------------------
    // 6.  Same gradient check at stride=2 (different code path through
    //     im2col / col2im -- input pixels are no longer covered by
    //     overlapping receptive fields).
    // ----------------------------------------------------------------
    {
        const std::size_t N = 1, in_C = 2, out_C = 2;
        const std::size_t K = 3, H = 5, W = 5, pad = 0;

        Conv2D<double> conv(in_C, out_C, K, /*stride=*/2, pad, /*seed=*/99u);

        Tensor4D<double> X(N, in_C, H, W);
        X.randomizeNormal(0.0, 1.0, 7u);

        Tensor4D<double> Y = conv.forward(X);
        Tensor4D<double> dY(Y.N(), Y.C(), Y.H(), Y.W());
        dY.randomizeNormal(0.0, 1.0, 8u);

        Tensor4D<double> dX_an = conv.backward(dY);

        const double eps     = 1e-5;
        const double tol_rel = 1e-6;
        const double tol_abs = 1e-8;
        auto ok = [&](double num, double an) {
            const double diff = std::fabs(num - an);
            if (diff <= tol_abs) return true;
            const double denom = std::max(std::fabs(num), std::fabs(an));
            return diff / denom <= tol_rel;
        };

        bool dX_ok = true;
        for (std::size_t n = 0; n < N && dX_ok; ++n)
        for (std::size_t c = 0; c < in_C && dX_ok; ++c)
        for (std::size_t h = 0; h < H && dX_ok; ++h)
        for (std::size_t w = 0; w < W && dX_ok; ++w) {
            const double saved = X(n, c, h, w);
            X(n, c, h, w) = saved + eps;
            Tensor4D<double> Yp = conv.forward(X);
            X(n, c, h, w) = saved - eps;
            Tensor4D<double> Ym = conv.forward(X);
            X(n, c, h, w) = saved;
            const double num = (dot(dY, Yp) - dot(dY, Ym)) / (2.0 * eps);
            const double an  = dX_an(n, c, h, w);
            if (!ok(num, an)) dX_ok = false;
        }
        check(dX_ok, "Conv2D: dX gradient correct at stride=2");
    }

    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
