#pragma once
//
// Pitch.h  --  note names and fundamental-frequency refinement.
//
//   note_to_hz("A4")  ->  440.0         (equal temperament, A4 = 440 Hz)
//
//   refine_f0(samples, sample_rate, nominal_hz, hop, n_frames)
//       ->  one fundamental frequency per control frame
//
// WHY REFINE.  A harmonic synthesiser places partial k at exactly k * f0.
// If f0 is off by 10 cents (0.6%), partial 40 is off by the same 0.6% --
// which at 17 kHz is 100 Hz, several FFT bins away from where the real
// partial is.  The note name on a sample tells us the pitch only to the
// nearest semitone; players tune a little sharp or flat and add vibrato.
// So we start from the nominal pitch and, frame by frame, search a small
// range around it for the fundamental whose harmonics best line up with
// the peaks of the measured spectrum.
//
// METHOD (harmonic summation).  For each frame take a zero-padded FFT and,
// for each candidate f in nominal * 2^(c/1200), c = -range..+range cents,
// add up the spectrum's magnitude at f, 2f, 3f, ...  The candidate with
// the largest sum wins.  Because the search is confined to less than a
// semitone around a known pitch there is no octave ambiguity, which is the
// hard part of general pitch tracking and the reason we do not need a
// full tracker here.  Quiet frames, where the estimate is meaningless,
// take the median of the confident ones; a short median filter removes
// isolated outliers.
//
#include "FFT.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

// "A4" -> 440, "C#3" -> 138.59, "Bb2" -> 116.54.
inline double note_to_hz(const std::string& name) {
    if (name.empty()) throw std::invalid_argument("note_to_hz: empty note name");
    static const int semis[7] = {9, 11, 0, 2, 4, 5, 7};      // A B C D E F G
    const char letter = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));
    if (letter < 'A' || letter > 'G')
        throw std::invalid_argument("note_to_hz: bad note letter in '" + name + "'");
    int semitone = semis[letter - 'A'];
    std::size_t i = 1;
    while (i < name.size() && (name[i] == '#' || name[i] == 'b')) {
        semitone += (name[i] == '#') ? 1 : -1;
        ++i;
    }
    if (i >= name.size())
        throw std::invalid_argument("note_to_hz: missing octave in '" + name + "'");
    const int octave = std::stoi(name.substr(i));
    const int midi = 12 * (octave + 1) + semitone;           // C4 = 60, A4 = 69
    return 440.0 * std::pow(2.0, (midi - 69) / 12.0);
}

// One refined fundamental (Hz) per control frame; frame t is centred on
// sample t * hop.
template <typename T>
std::vector<T> refine_f0(const std::vector<T>& samples, T sample_rate,
                         T nominal_hz, std::size_t hop, std::size_t n_frames,
                         T range_cents = T(60), std::size_t window = 4096,
                         std::size_t fft_size = 16384, std::size_t max_harmonics = 20)
{
    if (!detail::is_power_of_two(fft_size) || window > fft_size)
        throw std::invalid_argument("refine_f0: bad window / fft_size");

    std::vector<T> win(window);
    for (std::size_t i = 0; i < window; ++i)
        win[i] = T(0.5) * (T(1) - std::cos(T(2) * detail::pi_v<T> * T(i) / T(window)));

    const T nyquist = sample_rate * T(0.5);
    const T bin_hz  = sample_rate / T(fft_size);
    const int steps = static_cast<int>(range_cents);          // 1-cent grid

    std::vector<T> f0(n_frames, nominal_hz), energy(n_frames, T(0));
    std::vector<std::complex<T>> buf(fft_size);
    std::vector<T> mag(fft_size / 2 + 1);

    for (std::size_t t = 0; t < n_frames; ++t) {
        const long start = long(t * hop) - long(window / 2);
        T e = T(0);
        for (std::size_t i = 0; i < fft_size; ++i) buf[i] = std::complex<T>(T(0), T(0));
        for (std::size_t i = 0; i < window; ++i) {
            const long idx = start + long(i);
            if (idx < 0 || idx >= long(samples.size())) continue;
            const T v = samples[std::size_t(idx)] * win[i];
            buf[i] = std::complex<T>(v, T(0));
            e += v * v;
        }
        energy[t] = e;
        if (e <= T(0)) continue;
        fft(buf);
        for (std::size_t k = 0; k < mag.size(); ++k) mag[k] = std::abs(buf[k]);

        T best_score = T(-1), best_f = nominal_hz;
        for (int c = -steps; c <= steps; ++c) {
            const T f = nominal_hz * std::pow(T(2), T(c) / T(1200));
            T score = T(0);
            for (std::size_t h = 1; h <= max_harmonics; ++h) {
                const T fh = f * T(h);
                if (fh >= nyquist) break;
                const T pos = fh / bin_hz;
                const std::size_t k = static_cast<std::size_t>(pos);
                if (k + 1 >= mag.size()) break;
                const T fr = pos - T(k);
                score += (T(1) - fr) * mag[k] + fr * mag[k + 1];
            }
            if (score > best_score) { best_score = score; best_f = f; }
        }
        f0[t] = best_f;
    }

    // Frames with less than 1% of the loudest frame's energy are unreliable.
    const T e_max = *std::max_element(energy.begin(), energy.end());
    std::vector<T> confident;
    for (std::size_t t = 0; t < n_frames; ++t)
        if (energy[t] > T(0.01) * e_max) confident.push_back(f0[t]);
    T fallback = nominal_hz;
    if (!confident.empty()) {
        std::nth_element(confident.begin(),
                         confident.begin() + confident.size() / 2, confident.end());
        fallback = confident[confident.size() / 2];
    }
    for (std::size_t t = 0; t < n_frames; ++t)
        if (!(energy[t] > T(0.01) * e_max)) f0[t] = fallback;

    // 5-point median filter.
    std::vector<T> out(n_frames);
    for (std::size_t t = 0; t < n_frames; ++t) {
        T w[5];
        for (int d = -2; d <= 2; ++d) {
            long j = long(t) + d;
            if (j < 0) j = 0;
            if (j >= long(n_frames)) j = long(n_frames) - 1;
            w[d + 2] = f0[std::size_t(j)];
        }
        std::sort(w, w + 5);
        out[t] = w[2];
    }
    return out;
}

} // namespace weft
