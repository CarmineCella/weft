// tests/test_tensor4d.cpp
//
// Tensor4D unit tests.  The most important property to verify is that
// element access (n, c, h, w) and the flat NCHW layout agree -- if those
// disagree, every conv layer built on top will quietly produce garbage.

#include "../src/Tensor4D.h"

#include <iostream>
#include <stdexcept>
#include <set>

using namespace weft;

static int passed = 0, total = 0;

static void check(bool cond, const char* name) {
    ++total;
    if (cond) { ++passed; }
    else      { std::cout << "FAIL: " << name << "\n"; }
}

template <typename T>
static void close(T a, T b, T tol, const char* name) {
    ++total;
    const T d = a - b;
    if ((d < 0 ? -d : d) <= tol) { ++passed; }
    else { std::cout << "FAIL: " << name
                     << "  (got " << a << ", expected " << b << ")\n"; }
}

int main() {
    // ----------------------------------------------------------------
    // 1.  Default constructor: empty tensor.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X;
        check(X.N() == 0 && X.C() == 0 && X.H() == 0 && X.W() == 0,
              "default ctor: all dims 0");
        check(X.size() == 0, "default ctor: size 0");
    }

    // ----------------------------------------------------------------
    // 2.  Shaped constructor: dimensions and total size.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(2, 3, 4, 5);
        check(X.N() == 2 && X.C() == 3 && X.H() == 4 && X.W() == 5,
              "shaped ctor: dims");
        check(X.size() == 2 * 3 * 4 * 5, "shaped ctor: size = N*C*H*W");
        // Default value is 0.
        check(X(0, 0, 0, 0) == 0.0f && X(1, 2, 3, 4) == 0.0f,
              "shaped ctor: default value 0");
    }

    // ----------------------------------------------------------------
    // 3.  Shaped constructor with initial value.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 2, 3, 4, 7.5f);
        check(X(0, 0, 0, 0) == 7.5f && X(0, 1, 2, 3) == 7.5f,
              "shaped ctor: initial value applied");
    }

    // ----------------------------------------------------------------
    // 4.  Element access: write at (n, c, h, w), read it back.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(2, 3, 4, 5);
        X(0, 0, 0, 0) = 1.0f;
        X(1, 2, 3, 4) = 42.0f;
        X(0, 1, 2, 3) = -3.5f;
        check(X(0, 0, 0, 0) == 1.0f,  "write/read corner (0,0,0,0)");
        check(X(1, 2, 3, 4) == 42.0f, "write/read corner (1,2,3,4)");
        close(X(0, 1, 2, 3), -3.5f, 1e-6f, "write/read interior");
    }

    // ----------------------------------------------------------------
    // 5.  THE CRITICAL TEST.  Flat NCHW index mapping:
    //         data[((n*C + c)*H + h)*W + w]
    //     means stepping w by 1 advances 1 in flat,
    //     stepping h by 1 advances W,
    //     stepping c by 1 advances H*W,
    //     stepping n by 1 advances C*H*W.
    //
    //     If this fails, every conv layer is silently broken.
    // ----------------------------------------------------------------
    {
        const std::size_t N = 2, C = 3, H = 4, W = 5;
        Tensor4D<float> X(N, C, H, W);

        // Mark (0,0,0,0) and verify it sits at flat index 0.
        X(0, 0, 0, 0) = 1.0f;
        check(X.data()[0] == 1.0f,
              "NCHW: (0,0,0,0) -> flat 0");

        // Stepping w by 1 -> flat advances by 1.
        X.fill(0.0f);
        X(0, 0, 0, 1) = 1.0f;
        check(X.data()[1] == 1.0f,
              "NCHW: (0,0,0,1) -> flat 1  (W is innermost)");

        // Stepping h by 1 -> flat advances by W.
        X.fill(0.0f);
        X(0, 0, 1, 0) = 1.0f;
        check(X.data()[W] == 1.0f,
              "NCHW: (0,0,1,0) -> flat W");

        // Stepping c by 1 -> flat advances by H*W.
        X.fill(0.0f);
        X(0, 1, 0, 0) = 1.0f;
        check(X.data()[H * W] == 1.0f,
              "NCHW: (0,1,0,0) -> flat H*W");

        // Stepping n by 1 -> flat advances by C*H*W.
        X.fill(0.0f);
        X(1, 0, 0, 0) = 1.0f;
        check(X.data()[C * H * W] == 1.0f,
              "NCHW: (1,0,0,0) -> flat C*H*W");
    }

    // ----------------------------------------------------------------
    // 6.  fill() overwrites every element.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(2, 2, 3, 3, 1.0f);
        X.fill(-1.0f);
        bool all_neg_one = true;
        for (std::size_t k = 0; k < X.size(); ++k)
            if (X.data()[k] != -1.0f) { all_neg_one = false; break; }
        check(all_neg_one, "fill(-1) sets every element");
    }

    // ----------------------------------------------------------------
    // 7.  randomizeNormal produces a non-degenerate distribution.
    //     A reasonable proxy: at least 90% of values are distinct.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(2, 2, 3, 3);
        X.randomizeNormal(0.0f, 1.0f, 12345u);
        std::set<float> distinct;
        for (std::size_t k = 0; k < X.size(); ++k) distinct.insert(X.data()[k]);
        check(distinct.size() * 10 >= X.size() * 9,
              "randomizeNormal: most values distinct");
    }

    // ----------------------------------------------------------------
    // 8.  randomizeNormal is deterministic given a fixed seed.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> A(1, 1, 4, 4), B(1, 1, 4, 4);
        A.randomizeNormal(0.0f, 1.0f, 7u);
        B.randomizeNormal(0.0f, 1.0f, 7u);
        bool same = true;
        for (std::size_t k = 0; k < A.size(); ++k)
            if (A.data()[k] != B.data()[k]) { same = false; break; }
        check(same, "randomizeNormal: same seed -> same values");
    }

    // ----------------------------------------------------------------
    // 9.  apply() / applyInPlace() walk every element.
    //     This is how ReLU4D will be implemented.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> X(1, 2, 2, 2, 3.0f);
        Tensor4D<float> Y = X.apply([](float v){ return v * v; });
        check(Y(0, 0, 0, 0) == 9.0f && Y(0, 1, 1, 1) == 9.0f,
              "apply: f applied elementwise (returns new tensor)");
        // Original unchanged.
        check(X(0, 0, 0, 0) == 3.0f, "apply: original unchanged");

        X.applyInPlace([](float v){ return v + 1.0f; });
        check(X(0, 1, 1, 1) == 4.0f, "applyInPlace: modifies tensor");
    }

    // ----------------------------------------------------------------
    // 10. hadamard: element-wise product of same-shape tensors.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> A(1, 1, 2, 2, 2.0f);
        Tensor4D<float> B(1, 1, 2, 2, 5.0f);
        Tensor4D<float> C = hadamard(A, B);
        check(C(0, 0, 0, 0) == 10.0f && C(0, 0, 1, 1) == 10.0f,
              "hadamard: element-wise product");
    }

    // ----------------------------------------------------------------
    // 11. hadamard: shape mismatch must throw.
    // ----------------------------------------------------------------
    {
        Tensor4D<float> A(1, 2, 3, 4), B(2, 2, 3, 4);
        bool threw = false;
        try { (void)hadamard(A, B); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "hadamard: shape mismatch throws");
    }

    // ----------------------------------------------------------------
    std::cout << passed << " / " << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
