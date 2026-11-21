#pragma once
//
// FilteredNoise.h  --  white noise shaped by a time-varying filter.
//
// This is the "noise" half of a harmonic-plus-noise synthesiser: the
// breath of a flute, the bow noise of a cello, the hiss of an attack.
// None of that is periodic, so sinusoids cannot make it; filtered noise
// can.
//
// Controls:
//
//   M[b, t]    gain of frequency band b at control frame t      (B x T)
//
// The B bands are spread evenly from 0 Hz to the Nyquist frequency.  The
// gain at an FFT bin is the linear interpolation of the two nearest band
// gains, so M describes a smooth spectral envelope with B control points.
//
// Algorithm (a short-time Fourier filter):
//
//   1. Take a fixed white-noise signal (seeded, so the layer is
//      deterministic) and cut it into overlapping windowed frames;
//      FFT each frame.                                  -> X_t[k]
//   2. Multiply frame t's spectrum by the interpolated gains G_t[k].
//   3. Inverse-FFT, window again, overlap-add.           -> y[n]
//
// Control frame t is centred on sample t * hop, the same convention as
// HarmonicSynth, with fft_size = 2 * hop.  Analysis and synthesis both use
// a square-root Hann window, whose product is a Hann window; at 50%
// overlap Hann windows sum to exactly 1, so with all gains equal to 1 the
// output is the input noise unchanged.
//
// Like HarmonicSynth, the output is LINEAR in the controls, which makes
// the backward pass a straightforward adjoint: for the full-spectrum gain
//
//   dL/dG_t[k] = (1/N) Re( X_t[k] * conj( FFT(w * dy_t)[k] ) )
//
// where dy_t is the slice of the upstream gradient under frame t.  Bins k
// and N-k carry the same gain, so their gradients add; the band gradient
// is then the interpolation weights applied in reverse.
//
#include "FFT.h"
#include "Matrix.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace weft {

template <typename T = float>
class FilteredNoise {
public:
    // hop        samples between control frames; the FFT size is 2 * hop
    //            and must be a power of two
    // n_bands    number of gain control points across the spectrum (>= 2)
    FilteredNoise(std::size_t hop, std::size_t n_bands, unsigned seed = 1)
        : hop_(hop), N_(2 * hop), B_(n_bands), seed_(seed), window_(2 * hop)
    {
        if (!detail::is_power_of_two(N_))
            throw std::invalid_argument("FilteredNoise: 2 * hop must be a power of two");
        if (n_bands < 2)
            throw std::invalid_argument("FilteredNoise: need at least 2 bands");

        // Square-root (periodic) Hann.
        for (std::size_t i = 0; i < N_; ++i)
            window_[i] = std::sqrt(T(0.5) * (T(1) - std::cos(
                T(2) * detail::pi_v<T> * T(i) / T(N_))));

        // Bin k sits at position p = k / (N/2) * (B-1) on the band axis;
        // its gain interpolates bands floor(p) and floor(p)+1.
        const std::size_t half = N_ / 2;
        lo_.resize(half + 1);
        frac_.resize(half + 1);
        for (std::size_t k = 0; k <= half; ++k) {
            const T p = T(k) / T(half) * T(B_ - 1);
            std::size_t b = static_cast<std::size_t>(p);
            if (b >= B_ - 1) b = B_ - 2;
            lo_[k]   = b;
            frac_[k] = p - T(b);
        }
    }

    // M: (B x T) band gains.  Returns n_samples of audio.
    std::vector<T> forward(const Matrix<T>& M, std::size_t n_samples) {
        if (M.rows() != B_)
            throw std::invalid_argument("FilteredNoise::forward: M.rows() != n_bands");
        const std::size_t Tn = M.cols();
        if (Tn < (n_samples + hop_ - 1) / hop_ + 1)
            throw std::invalid_argument("FilteredNoise::forward: not enough control frames");

        prepare_noise(Tn);
        n_  = n_samples;
        Tn_ = Tn;

        const std::size_t half = N_ / 2;
        std::vector<T> y(n_, T(0));
        std::vector<std::complex<T>> buf(N_);
        for (std::size_t t = 0; t < Tn; ++t) {
            const std::complex<T>* X = &spectra_[t * N_];
            for (std::size_t k = 0; k <= half; ++k) {
                const T g = (T(1) - frac_[k]) * M(lo_[k], t)
                          + frac_[k] * M(lo_[k] + 1, t);
                buf[k] = X[k] * g;
                if (k != 0 && k != half) buf[N_ - k] = X[N_ - k] * g;
            }
            ifft(buf);
            // Frame t covers samples [t*hop - hop, t*hop + hop).
            for (std::size_t i = 0; i < N_; ++i) {
                const long idx = long(t * hop_) + long(i) - long(hop_);
                if (idx < 0 || idx >= long(n_)) continue;
                y[std::size_t(idx)] += buf[i].real() * window_[i];
            }
        }
        return y;
    }

    // dy: gradient w.r.t. the audio of the last forward.  Returns dM (B x T).
    Matrix<T> backward(const std::vector<T>& dy) const {
        if (dy.size() != n_)
            throw std::invalid_argument("FilteredNoise::backward: dy size mismatch");
        const std::size_t half = N_ / 2;
        Matrix<T> dM(B_, Tn_, T(0));
        std::vector<std::complex<T>> buf(N_);
        const T inv_N = T(1) / T(N_);
        for (std::size_t t = 0; t < Tn_; ++t) {
            bool any = false;
            for (std::size_t i = 0; i < N_; ++i) {
                const long idx = long(t * hop_) + long(i) - long(hop_);
                T v = T(0);
                if (idx >= 0 && idx < long(n_)) v = dy[std::size_t(idx)] * window_[i];
                if (v != T(0)) any = true;
                buf[i] = std::complex<T>(v, T(0));
            }
            if (!any) continue;
            fft(buf);
            const std::complex<T>* X = &spectra_[t * N_];
            for (std::size_t k = 0; k <= half; ++k) {
                T g = inv_N * (X[k] * std::conj(buf[k])).real();
                if (k != 0 && k != half) g *= T(2);       // bins k and N-k
                dM(lo_[k],     t) += (T(1) - frac_[k]) * g;
                dM(lo_[k] + 1, t) += frac_[k] * g;
            }
        }
        return dM;
    }

    std::size_t hop()     const { return hop_; }
    std::size_t n_bands() const { return B_; }

private:
    // Generate the noise and its frame spectra for Tn frames (cached; only
    // redone when more frames are needed than last time).
    void prepare_noise(std::size_t Tn) {
        if (Tn <= frames_ready_) return;
        const std::size_t len = (Tn + 1) * hop_;        // frame t starts at t*hop
        std::mt19937 gen(seed_);
        std::uniform_real_distribution<T> dist(T(-1), T(1));
        std::vector<T> noise(len);
        for (T& v : noise) v = dist(gen);

        spectra_.assign(Tn * N_, std::complex<T>(T(0), T(0)));
        std::vector<std::complex<T>> buf(N_);
        for (std::size_t t = 0; t < Tn; ++t) {
            for (std::size_t i = 0; i < N_; ++i)
                buf[i] = std::complex<T>(noise[t * hop_ + i] * window_[i], T(0));
            fft(buf);
            for (std::size_t i = 0; i < N_; ++i) spectra_[t * N_ + i] = buf[i];
        }
        frames_ready_ = Tn;
    }

    std::size_t hop_, N_, B_;
    unsigned    seed_;
    std::vector<T>           window_;
    std::vector<std::size_t> lo_;      // lower band index for each bin
    std::vector<T>           frac_;    // interpolation fraction for each bin
    std::vector<std::complex<T>> spectra_;   // (frames x N) noise spectra
    std::size_t frames_ready_ = 0;
    std::size_t n_ = 0, Tn_ = 0;
};

} // namespace weft
