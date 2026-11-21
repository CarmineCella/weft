#pragma once
//
// HarmonicSynth.h  --  a bank of sinusoids at integer multiples of a
//                      fundamental, with time-varying amplitudes.
//
// This is the "harmonic" half of a harmonic-plus-noise synthesiser.  A
// pitched instrument tone is, to a good approximation, a sum of partials
// at f0, 2 f0, 3 f0, ...; what distinguishes a flute from a cello is how
// loud each partial is and how that changes over time.  So the synthesiser
// takes
//
//   f0[t]      fundamental frequency in Hz, one value per control frame
//   A[k, t]    amplitude of harmonic k+1 at control frame t      (K x T)
//
// and produces audio
//
//   y[n] = sum_k  a_k[n] * sin( (k+1) * phi[n] )
//
// where phi[n] = 2 pi * cumulative sum of f0[n] / sample_rate  is the
// running phase of the fundamental, and a_k[n], f0[n] are the control
// values linearly interpolated from the frame rate up to the sample rate.
// Control frame t sits at sample t * hop.
//
// Harmonics whose frequency reaches the Nyquist limit are silenced, sample
// by sample, so the output never aliases.
//
// WHY THIS IS EASY TO DIFFERENTIATE
// ---------------------------------
// The output is LINEAR in the amplitudes: y = sum_{k,t} A[k,t] * b_{k,t}[n]
// with basis signals  b_{k,t}[n] = w_t[n] * sin((k+1) phi[n]),  w_t being
// the triangular interpolation weight of frame t.  The gradient is then
// just a correlation of the upstream gradient with each basis signal:
//
//   dA[k, t] = sum_n  dy[n] * w_t[n] * sin((k+1) phi[n])
//
// No gradient flows to f0: pitch is an input here, not something learned.
//
// The sines are generated with the recurrence
//   sin((k+1) x) = 2 cos(x) sin(k x) - sin((k-1) x)
// which costs one multiply per harmonic instead of one sin() call.
//
#include "Matrix.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace weft {

template <typename T = float>
class HarmonicSynth {
public:
    HarmonicSynth(T sample_rate, std::size_t hop)
        : sr_(sample_rate), hop_(hop)
    {
        if (hop == 0) throw std::invalid_argument("HarmonicSynth: hop must be > 0");
    }

    // Number of control frames needed to cover n_samples of audio.
    static std::size_t frames_for(std::size_t n_samples, std::size_t hop) {
        return (n_samples + hop - 1) / hop + 1;
    }

    // f0: T control frames (Hz).   A: (K x T) harmonic amplitudes.
    // Returns n_samples of audio.
    std::vector<T> forward(const std::vector<T>& f0, const Matrix<T>& A,
                           std::size_t n_samples)
    {
        const std::size_t Tn = f0.size();
        if (A.cols() != Tn)
            throw std::invalid_argument("HarmonicSynth::forward: A.cols() != f0.size()");
        if (Tn < frames_for(n_samples, hop_))
            throw std::invalid_argument("HarmonicSynth::forward: not enough control frames");

        K_ = A.rows();
        Tn_ = Tn;
        n_  = n_samples;

        // Per-sample fundamental and running phase.  The phase is
        // accumulated in double whatever T is (and wrapped to stay below
        // 2 pi): its rounding error is amplified by the harmonic number, up
        // to 100 times for the top harmonics.
        f0s_.resize(n_);
        phase_.resize(n_);
        const double two_pi = 6.283185307179586476925;
        double phi = 0.0;
        for (std::size_t n = 0; n < n_; ++n) {
            const std::size_t t0 = n / hop_;
            const double frac = double(n % hop_) / double(hop_);
            const double f = (1.0 - frac) * f0[t0] + frac * f0[t0 + 1];
            f0s_[n]   = f;
            phase_[n] = phi;
            phi += two_pi * f / double(sr_);
            if (phi > two_pi) phi -= two_pi;      // keep it small; sin is periodic
        }

        std::vector<T> y(n_, T(0));
        const double nyquist = 0.5 * double(sr_);
        for (std::size_t n = 0; n < n_; ++n) {
            const std::size_t t0 = n / hop_;
            const T frac = T(n % hop_) / T(hop_);
            const double c2 = 2.0 * std::cos(phase_[n]);
            double s_prev = 0.0;                  // sin(0 * phi)
            double s_cur  = std::sin(phase_[n]);  // sin(1 * phi)
            double acc = 0.0;
            for (std::size_t k = 0; k < K_; ++k) {
                if (double(k + 1) * f0s_[n] >= nyquist) break;
                const T a = (T(1) - frac) * A(k, t0) + frac * A(k, t0 + 1);
                acc += double(a) * s_cur;
                const double s_next = c2 * s_cur - s_prev;
                s_prev = s_cur;
                s_cur  = s_next;
            }
            y[n] = T(acc);
        }
        return y;
    }

    // dy: gradient w.r.t. the audio of the last forward.  Returns dA (K x T).
    Matrix<T> backward(const std::vector<T>& dy) const {
        if (dy.size() != n_)
            throw std::invalid_argument("HarmonicSynth::backward: dy size mismatch");
        Matrix<T> dA(K_, Tn_, T(0));
        const double nyquist = 0.5 * double(sr_);
        for (std::size_t n = 0; n < n_; ++n) {
            const T g = dy[n];
            if (g == T(0)) continue;
            const std::size_t t0 = n / hop_;
            const T frac = T(n % hop_) / T(hop_);
            const T g0 = g * (T(1) - frac);
            const T g1 = g * frac;
            const double c2 = 2.0 * std::cos(phase_[n]);
            double s_prev = 0.0;
            double s_cur  = std::sin(phase_[n]);
            for (std::size_t k = 0; k < K_; ++k) {
                if (double(k + 1) * f0s_[n] >= nyquist) break;
                dA(k, t0)     += g0 * T(s_cur);
                dA(k, t0 + 1) += g1 * T(s_cur);
                const double s_next = c2 * s_cur - s_prev;
                s_prev = s_cur;
                s_cur  = s_next;
            }
        }
        return dA;
    }

    T           sample_rate() const { return sr_; }
    std::size_t hop()         const { return hop_; }

private:
    T           sr_;
    std::size_t hop_;
    std::size_t K_ = 0, Tn_ = 0, n_ = 0;
    std::vector<double> f0s_;     // per-sample fundamental, cached for backward
    std::vector<double> phase_;   // per-sample fundamental phase
};

} // namespace weft
