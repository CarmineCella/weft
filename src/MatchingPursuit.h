#pragma once
//
// MatchingPursuit.h  --  decompose a sound into a list of time-frequency
//                        atoms, and rebuild it from that list.
//
// THE IDEA
// --------
// A spectrogram describes a sound with a fixed grid: every frame, every
// bin, one number each, whether anything is happening there or not.
// Matching pursuit describes it with a LIST: "a 93 ms tone at 440 Hz
// starting at 1.20 s with this amplitude; a 3 ms click at 2.05 s; ...".
// Each entry is an *atom*, a short windowed sinusoid, and the sound is
// approximated by their sum.  A few thousand atoms can stand for hundreds
// of thousands of samples, and each one means something.
//
// THE DICTIONARY
// --------------
// The atoms come from a Gabor dictionary: a Hann window of length N
// multiplied by a cosine,
//
//     g[n] = w_N[n] * cos(2 pi k n / N + phase),   scaled to unit energy
//
// for several window lengths N at once (by default 128, 512, 2048 and
// 8192 samples: 3 ms to 186 ms at 44.1 kHz), every frequency bin k of each
// length, and time positions a quarter-window apart.  Short atoms can
// place a transient precisely; long atoms can describe a steady partial
// with one entry.  No single-resolution transform can do both, and that
// is the reason for a multi-scale dictionary.
//
// THE ALGORITHM (Mallat & Zhang, 1993)
// ------------------------------------
// Greedy and simple:
//
//     residual = signal
//     repeat:
//         find the atom g that correlates best with the residual
//         a = <residual, g>
//         residual -= a * g
//         record (g, a)
//
// Because g has unit energy, every step removes exactly a^2 of energy, so
//
//     ||signal||^2 = ||residual||^2 + sum of a^2 over all atoms
//
// holds at every step (the tests check it), and the residual can only
// shrink.
//
// MAKING IT FAST
// --------------
// "Find the best atom" is a search over millions of candidates, but the
// correlation of the residual with every atom of one length is exactly a
// short-time Fourier transform of the residual: one FFT per frame gives
// the correlations for all frequencies and, through its phase, the best
// phase for each.  And subtracting an atom changes the residual only
// under that atom, so after each step we redo only the frames that
// overlap it.  For every frame we remember its best bin; picking the next
// atom is then a scan over frames, not over all coefficients.
//
// One subtlety: the FFT gives the correlation with a COMPLEX exponential.
// For a real cosine it is a very good guide to which atom is best and
// what its phase should be, but not an exact amplitude.  So once the atom
// is chosen we build it in the time domain and compute a = <residual, g>
// exactly.  That keeps the energy identity above exact.
//
#include "FFT.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace weft {

// One atom of a decomposition.
struct Atom {
    std::uint32_t scale = 0;   // index into the dictionary's window lengths
    std::int32_t  start = 0;   // first sample of the atom (may be negative)
    std::uint32_t bin   = 0;   // frequency bin: f = bin * sample_rate / N
    float         amp   = 0;   // coefficient of the unit-energy atom
    float         phase = 0;   // radians, at the first sample of the atom
};

template <typename T = float>
class MatchingPursuit {
public:
    explicit MatchingPursuit(std::vector<std::size_t> window_lengths =
                                 {128, 512, 2048, 8192})
        : sizes_(std::move(window_lengths))
    {
        if (sizes_.empty())
            throw std::invalid_argument("MatchingPursuit: no window lengths");
        for (std::size_t N : sizes_) {
            if (!detail::is_power_of_two(N) || N < 16)
                throw std::invalid_argument(
                    "MatchingPursuit: window lengths must be powers of two >= 16");
            std::vector<T> w(N);
            double e = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                w[i] = T(0.5) * (T(1) - std::cos(T(2) * detail::pi_v<T> * T(i) / T(N)));
                e += double(w[i]) * w[i];
            }
            windows_.push_back(std::move(w));
            // A unit-energy atom w * cos(...) has energy about ||w||^2 / 2,
            // so |<r, g>| is about |STFT coefficient| * sqrt(2) / ||w||.
            score_scale_.push_back(T(std::sqrt(2.0 / e)));
        }
        max_size_ = *std::max_element(sizes_.begin(), sizes_.end());
    }

    const std::vector<std::size_t>& window_lengths() const { return sizes_; }

    // Frequency in Hz of an atom.
    T frequency(const Atom& a, T sample_rate) const {
        return T(a.bin) * sample_rate / T(sizes_[a.scale]);
    }
    // Centre of an atom in samples.
    T centre(const Atom& a) const {
        return T(a.start) + T(0.5) * T(sizes_[a.scale]);
    }

    // The unit-energy waveform of one atom (length = its window length).
    std::vector<T> waveform(std::size_t scale, std::size_t bin, T phase) const {
        const std::size_t N = sizes_[scale];
        const std::vector<T>& w = windows_[scale];
        std::vector<T> g(N);
        const double step = 2.0 * 3.14159265358979323846 * double(bin) / double(N);
        double e = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            g[i] = w[i] * T(std::cos(step * double(i) + double(phase)));
            e += double(g[i]) * g[i];
        }
        const T inv = (e > 0.0) ? T(1.0 / std::sqrt(e)) : T(0);
        for (T& v : g) v *= inv;
        return g;
    }

    // Decompose `signal` into at most `max_atoms` atoms, stopping early
    // when the residual energy falls below `stop_ratio` of the signal's.
    // If `energy_trace` is given it receives the residual energy after
    // every atom (energy_trace[0] is the signal's own energy).
    std::vector<Atom> decompose(const std::vector<T>& signal, std::size_t max_atoms,
                                T stop_ratio = T(0),
                                std::vector<double>* energy_trace = nullptr,
                                std::vector<T>* residual_out = nullptr) const
    {
        // Work on a copy padded with silence on both sides, so that atoms
        // may overhang the ends of the signal.
        const std::size_t pad = max_size_;
        const std::size_t len = signal.size() + 2 * pad;
        std::vector<T> r(len, T(0));
        std::copy(signal.begin(), signal.end(), r.begin() + long(pad));

        double energy = 0.0;
        for (T v : signal) energy += double(v) * v;
        const double energy0 = energy;
        if (energy_trace) { energy_trace->clear(); energy_trace->push_back(energy); }

        // Per scale: the best bin of every frame, its score and its phase.
        struct Frames {
            std::size_t hop = 0, count = 0;
            std::vector<T>             score;
            std::vector<std::uint32_t> bin;
            std::vector<T>             phase;
        };
        std::vector<Frames> frames(sizes_.size());
        std::vector<std::complex<T>> buf;

        auto analyse = [&](std::size_t s, std::size_t f) {
            const std::size_t N = sizes_[s];
            const std::vector<T>& w = windows_[s];
            const std::size_t start = f * frames[s].hop;
            buf.resize(N);
            bool silent = true;
            for (std::size_t i = 0; i < N; ++i) {
                const T v = r[start + i];
                if (v != T(0)) silent = false;
                buf[i] = std::complex<T>(v * w[i], T(0));
            }
            if (silent) { frames[s].score[f] = T(0); return; }
            fft(buf);
            // Bins 0 and N/2 are skipped: there the cosine and its mirror
            // image coincide and the FFT is a poor guide to the best phase.
            T best = T(-1); std::size_t best_k = 1;
            for (std::size_t k = 1; k < N / 2; ++k) {
                const T m = std::norm(buf[k]);
                if (m > best) { best = m; best_k = k; }
            }
            frames[s].score[f] = std::sqrt(best) * score_scale_[s];
            frames[s].bin[f]   = static_cast<std::uint32_t>(best_k);
            frames[s].phase[f] = std::arg(buf[best_k]);
        };

        for (std::size_t s = 0; s < sizes_.size(); ++s) {
            Frames& F = frames[s];
            F.hop   = sizes_[s] / 4;
            F.count = (len - sizes_[s]) / F.hop + 1;
            F.score.assign(F.count, T(0));
            F.bin.assign(F.count, 0);
            F.phase.assign(F.count, T(0));
            for (std::size_t f = 0; f < F.count; ++f) analyse(s, f);
        }

        std::vector<Atom> atoms;
        atoms.reserve(max_atoms);
        while (atoms.size() < max_atoms && energy > double(stop_ratio) * energy0) {
            // The best frame over all scales.
            T best = T(0); std::size_t bs = 0, bf = 0;
            for (std::size_t s = 0; s < sizes_.size(); ++s) {
                const Frames& F = frames[s];
                for (std::size_t f = 0; f < F.count; ++f)
                    if (F.score[f] > best) { best = F.score[f]; bs = s; bf = f; }
            }
            if (best <= T(0)) break;                       // nothing left

            const std::size_t N     = sizes_[bs];
            const std::size_t start = bf * frames[bs].hop;
            const std::vector<T> g  = waveform(bs, frames[bs].bin[bf], frames[bs].phase[bf]);

            // Exact projection in the time domain.
            double a = 0.0;
            for (std::size_t i = 0; i < N; ++i) a += double(r[start + i]) * g[i];
            if (a == 0.0) { frames[bs].score[bf] = T(0); continue; }
            for (std::size_t i = 0; i < N; ++i) r[start + i] -= T(a) * g[i];
            energy -= a * a;
            if (energy < 0.0) energy = 0.0;

            Atom atom;
            atom.scale = static_cast<std::uint32_t>(bs);
            atom.start = static_cast<std::int32_t>(long(start) - long(pad));
            atom.bin   = frames[bs].bin[bf];
            atom.amp   = static_cast<float>(a);
            atom.phase = static_cast<float>(frames[bs].phase[bf]);
            atoms.push_back(atom);
            if (energy_trace) energy_trace->push_back(energy);

            // Only frames that overlap [start, start + N) have changed.
            for (std::size_t s = 0; s < sizes_.size(); ++s) {
                const Frames& F = frames[s];
                const long Ns = long(sizes_[s]), hop = long(F.hop);
                long f0 = (long(start) - Ns + hop) / hop;         // first overlapping frame
                if (f0 < 0) f0 = 0;
                long f1 = (long(start) + long(N) - 1) / hop;       // last overlapping frame
                if (f1 >= long(F.count)) f1 = long(F.count) - 1;
                for (long f = f0; f <= f1; ++f) analyse(s, std::size_t(f));
            }
        }

        if (residual_out)
            residual_out->assign(r.begin() + long(pad),
                                 r.begin() + long(pad + signal.size()));
        return atoms;
    }

    // Rebuild a signal of n_samples from a list of atoms.
    std::vector<T> synthesize(const std::vector<Atom>& atoms, std::size_t n_samples) const {
        std::vector<T> y(n_samples, T(0));
        for (const Atom& a : atoms) {
            if (a.scale >= sizes_.size()) continue;
            const std::vector<T> g = waveform(a.scale, a.bin, T(a.phase));
            for (std::size_t i = 0; i < g.size(); ++i) {
                const long idx = long(a.start) + long(i);
                if (idx < 0 || idx >= long(n_samples)) continue;
                y[std::size_t(idx)] += T(a.amp) * g[i];
            }
        }
        return y;
    }

private:
    std::vector<std::size_t>    sizes_;
    std::vector<std::vector<T>> windows_;
    std::vector<T>              score_scale_;
    std::size_t                 max_size_ = 0;
};

} // namespace weft
