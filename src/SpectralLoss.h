#pragma once
//
// SpectralLoss.h  --  multi-scale spectral loss between two waveforms.
//
// Comparing two sounds sample by sample is the wrong test.  Two sine waves
// of the same frequency and loudness but different starting phase sound
// identical, yet their sample-wise difference can be as large as the
// signals themselves.  What the ear compares is closer to the *magnitude*
// of the short-time spectrum, so that is what this loss compares.
//
// For one FFT size N (Hann window, hop N/4):
//
//   S_y[k, f] = | STFT(y) |[k, f] / sum(window)     (same for the target)
//
//   L_N = mean over (k, f) of
//           | S_y - S_x |  +  alpha * | log(S_y + eps) - log(S_x + eps) |
//
// The linear term is dominated by the loud partials; the log term makes
// quiet components (the decay of a note, soft high harmonics) count too.
//
// One FFT size is a compromise: long windows resolve frequency but blur
// time, short windows do the opposite.  So we sum the loss over several
// sizes ("multi-scale"), by default 2048, 1024, 512, 256, 128 and 64.
//
// BACKWARD
// --------
// For one frame, Y[k] = sum_i y[i] w[i] exp(-2 pi j k i / N), so
//
//   d|Y[k]| / dy[i] = w[i] * Re( conj(Y[k]) / |Y[k]| * exp(-2 pi j k i / N) )
//
// Collect the per-bin loss gradients g[k] = dL/d|Y[k]| into
// C[k] = g[k] * conj(Y[k]) / |Y[k]| (zero outside bins 0..N/2); then the
// frame's gradient is  w[i] * Re( FFT(C)[i] ), and frames overlap-add into
// the gradient of the whole signal.  One FFT per frame, like the forward.
//
// Usage: set_target(x) once (its spectra are cached), then forward(y) /
// backward() as often as needed.
//
#include "FFT.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace weft {

template <typename T = float>
class SpectralLoss {
public:
    explicit SpectralLoss(std::vector<std::size_t> fft_sizes =
                              {2048, 1024, 512, 256, 128, 64},
                          T alpha = T(1), T eps = T(1e-5))
        : sizes_(std::move(fft_sizes)), alpha_(alpha), eps_(eps)
    {
        if (sizes_.empty())
            throw std::invalid_argument("SpectralLoss: no FFT sizes given");
        for (std::size_t N : sizes_) {
            if (!detail::is_power_of_two(N) || N < 4)
                throw std::invalid_argument(
                    "SpectralLoss: FFT sizes must be powers of two >= 4");
            Scale s;
            s.N   = N;
            s.hop = N / 4;
            s.window.resize(N);
            T sum = T(0);
            for (std::size_t i = 0; i < N; ++i) {
                s.window[i] = T(0.5) * (T(1) - std::cos(
                    T(2) * detail::pi_v<T> * T(i) / T(N)));
                sum += s.window[i];
            }
            s.norm = T(1) / sum;
            scales_.push_back(std::move(s));
        }
    }

    // Cache the target's magnitude spectra.
    void set_target(const std::vector<T>& x) {
        len_ = x.size();
        std::vector<std::complex<T>> buf;
        for (Scale& s : scales_) {
            if (len_ < s.N)
                throw std::invalid_argument("SpectralLoss: signal shorter than an FFT size");
            const std::size_t nb = s.N / 2 + 1;
            s.frames = (len_ - s.N) / s.hop + 1;
            s.target.assign(s.frames * nb, T(0));
            buf.resize(s.N);
            for (std::size_t f = 0; f < s.frames; ++f) {
                for (std::size_t i = 0; i < s.N; ++i)
                    buf[i] = std::complex<T>(x[f * s.hop + i] * s.window[i], T(0));
                fft(buf);
                for (std::size_t k = 0; k < nb; ++k)
                    s.target[f * nb + k] = std::abs(buf[k]) * s.norm;
            }
        }
    }

    // Loss of y against the cached target.  Caches what backward needs.
    T forward(const std::vector<T>& y) {
        if (y.size() != len_)
            throw std::invalid_argument("SpectralLoss::forward: length differs from target");
        T total = T(0);
        std::vector<std::complex<T>> buf;
        for (Scale& s : scales_) {
            const std::size_t nb = s.N / 2 + 1;
            s.spec.resize(s.frames * nb);
            buf.resize(s.N);
            T acc = T(0);
            for (std::size_t f = 0; f < s.frames; ++f) {
                for (std::size_t i = 0; i < s.N; ++i)
                    buf[i] = std::complex<T>(y[f * s.hop + i] * s.window[i], T(0));
                fft(buf);
                for (std::size_t k = 0; k < nb; ++k) {
                    s.spec[f * nb + k] = buf[k];
                    const T m  = std::abs(buf[k]) * s.norm;
                    const T mt = s.target[f * nb + k];
                    acc += std::fabs(m - mt)
                         + alpha_ * std::fabs(std::log(m + eps_) - std::log(mt + eps_));
                }
            }
            total += acc / T(s.frames * nb);
        }
        return total;
    }

    // dL/dy for the last forward.
    std::vector<T> backward() const {
        std::vector<T> dy(len_, T(0));
        std::vector<std::complex<T>> C;
        for (const Scale& s : scales_) {
            const std::size_t nb = s.N / 2 + 1;
            const T scale = T(1) / T(s.frames * nb);
            C.resize(s.N);
            for (std::size_t f = 0; f < s.frames; ++f) {
                for (std::size_t k = 0; k < s.N; ++k) C[k] = std::complex<T>(T(0), T(0));
                for (std::size_t k = 0; k < nb; ++k) {
                    const std::complex<T> Y = s.spec[f * nb + k];
                    const T mag = std::abs(Y);
                    if (mag == T(0)) continue;          // |.| not differentiable at 0
                    const T m  = mag * s.norm;
                    const T mt = s.target[f * nb + k];
                    const T d_lin = (m > mt) ? T(1) : (m < mt ? T(-1) : T(0));
                    const T ld    = std::log(m + eps_) - std::log(mt + eps_);
                    const T d_log = (ld > T(0)) ? T(1) : (ld < T(0) ? T(-1) : T(0));
                    const T g = scale * (d_lin + alpha_ * d_log / (m + eps_));
                    C[k] = (g * s.norm / mag) * std::conj(Y);
                }
                fft(C);
                for (std::size_t i = 0; i < s.N; ++i)
                    dy[f * s.hop + i] += s.window[i] * C[i].real();
            }
        }
        return dy;
    }

    const std::vector<std::size_t>& fft_sizes() const { return sizes_; }

private:
    struct Scale {
        std::size_t N = 0, hop = 0, frames = 0;
        T norm = T(1);
        std::vector<T> window;
        std::vector<T> target;                  // (frames x bins) magnitudes
        std::vector<std::complex<T>> spec;      // (frames x bins) last forward
    };
    std::vector<std::size_t> sizes_;
    std::vector<Scale>       scales_;
    T alpha_, eps_;
    std::size_t len_ = 0;
};

} // namespace weft
