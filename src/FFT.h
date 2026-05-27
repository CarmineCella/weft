#pragma once
//
// FFT.h -- iterative Cooley-Tukey radix-2 fast Fourier transform.
//
// fft(x) computes, in place,
//     X[k] = sum_{n=0..N-1}  x[n] * exp(-2*pi*i * k * n / N)
// for a complex input vector x of length N, where N must be a power of 2.
//
// ifft(x) computes the inverse,
//     x[n] = (1/N) * sum_{k=0..N-1}  X[k] * exp(+2*pi*i * k * n / N)
// so that ifft(fft(x)) == x up to floating-point error.
//
// Algorithm:  bit-reversal permutation, then log2(N) "butterfly" stages.
// Each butterfly combines a pair of complex values with a "twiddle"
// factor w = exp(-2*pi*i / len) at stage size `len`.  Total work is
// (N/2) * log2(N) complex multiplies -- O(N log N) instead of the naive
// O(N^2) sum.
//
// For real input (e.g. audio samples) lift to complex by copying samples
// into the real part and zeroing the imaginary part.  The output is then
// conjugate-symmetric: X[N-k] = conj(X[k]).  Only the first N/2 + 1 bins
// (DC, positive frequencies, and Nyquist) carry independent information.
//
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace weft {

namespace detail {

template <typename T>
inline constexpr T pi_v = T(3.14159265358979323846L);

// A power of two has exactly one bit set, so (n & (n - 1)) == 0.
// (Exception: zero also satisfies this, hence the separate n == 0 guard
// at the call site.)
inline bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

} // namespace detail

template <typename T>
void fft(std::vector<std::complex<T>>& x) {
    const std::size_t n = x.size();
    if (!detail::is_power_of_two(n))
        throw std::invalid_argument("FFT: size must be a power of 2");

    // ---- Bit-reversal permutation ----
    //
    // The butterfly stages below assume input is already in bit-reversed
    // order (so e.g. for N=8: indices 0..7 -> 0,4,2,6,1,5,3,7).  We do
    // this with an O(N) loop that tracks the bit-reversed index `j` as
    // `i` walks forward in natural order.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }

    // ---- Butterfly stages ----
    //
    // At stage `len`, we're combining adjacent length-(len/2) transforms
    // into length-len transforms.  Repeat for len = 2, 4, 8, ..., N.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const T angle = -T(2) * detail::pi_v<T> / T(len);
        const std::complex<T> wlen(std::cos(angle), std::sin(angle));

        for (std::size_t i = 0; i < n; i += len) {
            std::complex<T> w(T(1), T(0));
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<T> u = x[i + k];
                const std::complex<T> v = x[i + k + len / 2] * w;
                x[i + k]             = u + v;
                x[i + k + len / 2]   = u - v;
                w *= wlen;
            }
        }
    }
}

template <typename T>
void ifft(std::vector<std::complex<T>>& x) {
    // Standard trick: IFFT(x) = (1/N) * conj(FFT(conj(x))).
    // Saves writing a near-duplicate of fft() with the sign flipped.
    for (auto& v : x) v = std::conj(v);
    fft(x);
    const T inv_n = T(1) / T(x.size());
    for (auto& v : x) v = std::conj(v) * inv_n;
}

} // namespace weft
