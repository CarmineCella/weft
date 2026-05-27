// test_fft.cpp
//
// Tests for the Cooley-Tukey FFT.  Use doubles for the tighter tolerance
// so a numerical bug stands out from honest float rounding.
//
#include "FFT.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::complex;
using weft::fft;
using weft::ifft;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}
static bool close(complex<double> a, complex<double> b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

int main() {
    std::cout << "weft :: FFT tests\n";

    constexpr double PI = 3.14159265358979323846;

    // ---- 1. Size 1: identity (the trivial DFT) ----
    {
        std::vector<complex<double>> x = { {3.0, 2.0} };
        fft(x);
        check(close(x[0], complex<double>(3.0, 2.0)),
              "size 1: identity");
    }

    // ---- 2. Size 2: closed form  ----
    //   FFT([a, b]) = [a+b, a-b]
    {
        std::vector<complex<double>> x = { {1.0, 0.0}, {2.0, 0.0} };
        fft(x);
        check(close(x[0], {3.0, 0.0}) && close(x[1], {-1.0, 0.0}),
              "size 2: [a, b] -> [a+b, a-b]");
    }

    // ---- 3. Size 4: closed form ----
    //   FFT([1, 2, 3, 4]) = [10, -2+2i, -2, -2-2i]
    {
        std::vector<complex<double>> x = {
            {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}
        };
        fft(x);
        bool ok =
            close(x[0], {10.0,  0.0}) &&
            close(x[1], {-2.0,  2.0}) &&
            close(x[2], {-2.0,  0.0}) &&
            close(x[3], {-2.0, -2.0});
        check(ok, "size 4: matches hand-computed DFT");
    }

    // ---- 4. DC input ----
    //   x = [1, 1, ..., 1]  ->  X = [N, 0, 0, ..., 0]
    {
        const std::size_t N = 16;
        std::vector<complex<double>> x(N, {1.0, 0.0});
        fft(x);
        bool ok = close(x[0], {double(N), 0.0});
        for (std::size_t i = 1; i < N; ++i)
            if (!close(x[i], {0.0, 0.0})) ok = false;
        check(ok, "DC input: X[0] = N, all other bins = 0");
    }

    // ---- 5. Pure cosine ----
    //   x[n] = cos(2*pi*k0*n/N)  ->  |X[k0]| = |X[N-k0]| = N/2,
    //                                  all other bins ~ 0.
    {
        const std::size_t N  = 32;
        const std::size_t k0 = 5;
        std::vector<complex<double>> x(N);
        for (std::size_t n = 0; n < N; ++n)
            x[n] = { std::cos(2 * PI * k0 * n / N), 0.0 };
        fft(x);

        bool ok = true;
        ok &= close(std::abs(x[k0]),     N / 2.0, 1e-9);
        ok &= close(std::abs(x[N - k0]), N / 2.0, 1e-9);
        for (std::size_t i = 0; i < N; ++i) {
            if (i == k0 || i == N - k0) continue;
            if (!close(std::abs(x[i]), 0.0, 1e-9)) ok = false;
        }
        check(ok, "pure cosine: magnitude peaks at bins k0 and N-k0");
    }

    // ---- 6. Conjugate symmetry of real input ----
    //   For real x,  X[N-k] = conj(X[k]).
    {
        const std::size_t N = 16;
        std::vector<complex<double>> x(N);
        for (std::size_t n = 0; n < N; ++n)
            x[n] = { std::sin(0.3 * n) + 0.5 * std::cos(0.8 * n), 0.0 };
        fft(x);
        bool ok = true;
        for (std::size_t k = 1; k < N / 2; ++k)
            if (!close(x[N - k], std::conj(x[k]))) ok = false;
        check(ok, "real input: X[N-k] = conj(X[k])");
    }

    // ---- 7. Linearity ----
    //   FFT(a*x + b*y) == a*FFT(x) + b*FFT(y)
    {
        const std::size_t N = 8;
        std::vector<complex<double>> x(N), y(N);
        for (std::size_t i = 0; i < N; ++i) {
            x[i] = { double(i),       0.0           };
            y[i] = { 0.0,              0.5 * double(i) };
        }
        const double a = 2.0, b = -3.0;

        std::vector<complex<double>> combined(N);
        for (std::size_t i = 0; i < N; ++i)
            combined[i] = a * x[i] + b * y[i];

        auto Xt = x, Yt = y;
        fft(Xt); fft(Yt); fft(combined);

        bool ok = true;
        for (std::size_t i = 0; i < N; ++i)
            if (!close(combined[i], a * Xt[i] + b * Yt[i])) ok = false;
        check(ok, "linearity: FFT(ax + by) = a*FFT(x) + b*FFT(y)");
    }

    // ---- 8. Round trip on length 1024 ----
    //   ifft(fft(x)) == x  for arbitrary complex input.
    {
        const std::size_t N = 1024;
        std::vector<complex<double>> x(N);
        for (std::size_t i = 0; i < N; ++i)
            x[i] = { std::sin(i * 0.13), std::cos(i * 0.07) };
        auto orig = x;

        fft(x);
        ifft(x);

        bool ok = true;
        for (std::size_t i = 0; i < N; ++i)
            if (!close(x[i], orig[i], 1e-10)) ok = false;
        check(ok, "round trip ifft(fft(x)) == x  on N=1024");
    }

    // ---- 9. Non-power-of-2 throws ----
    {
        std::vector<complex<double>> x(3);
        bool threw = false;
        try { fft(x); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "size 3 throws invalid_argument");
    }

    // ---- 10. Size 0 throws ----
    {
        std::vector<complex<double>> x;
        bool threw = false;
        try { fft(x); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "size 0 throws invalid_argument");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
