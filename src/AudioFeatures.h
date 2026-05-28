#pragma once
//
// AudioFeatures.h -- feature extraction for audio classification.
//
// Two feature types, both producing a single fixed-length vector per
// input signal (energy-weighted across frames):
//
//   logmag_spectrum  -- log-magnitude FFT spectrum, length frame_size/2+1.
//                       Full bin-by-bin envelope; lots of detail, large
//                       feature vector.
//
//   mfcc             -- mel-frequency cepstral coefficients, length
//                       num_mfcc (default 13).  Compact summary of the
//                       spectral envelope on a perceptually-motivated
//                       scale, with a DCT to decorrelate.
//
// Both share the same framing skeleton:
//
//   1. Pre-compute a Hann window of length frame_size.
//   2. Slide the window across the signal with step hop_size.
//   3. For each frame: multiply by the window, compute the per-frame
//      feature, weight by the frame's energy (sum of squared samples).
//   4. Sum across frames and divide by total energy.
//
// "Energy-weighted" means quiet frames contribute less than loud ones,
// which is what we want -- the silent tail of a piano note shouldn't
// dilute the spectral character of its attack.
//
#include "FFT.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace weft {

namespace detail {

// Hann window: w[i] = 0.5 * (1 - cos(2 pi i / (N-1))).  Smooth taper to
// zero at both endpoints, suppresses spectral leakage from rectangular
// framing.
template <typename T>
std::vector<T> hann_window(std::size_t n) {
    std::vector<T> w(n);
    if (n <= 1) { if (n == 1) w[0] = T(1); return w; }
    const T scale = T(2) * pi_v<T> / T(n - 1);
    for (std::size_t i = 0; i < n; ++i)
        w[i] = T(0.5) * (T(1) - std::cos(scale * T(i)));
    return w;
}

// Magnitude spectrum of a single windowed frame, length N/2 + 1.
template <typename T>
std::vector<T> frame_magnitudes(const std::vector<T>& frame) {
    std::vector<std::complex<T>> buf(frame.size());
    for (std::size_t i = 0; i < frame.size(); ++i)
        buf[i] = std::complex<T>(frame[i], T(0));
    fft(buf);
    std::vector<T> mag(frame.size() / 2 + 1);
    for (std::size_t i = 0; i < mag.size(); ++i)
        mag[i] = std::abs(buf[i]);
    return mag;
}

// Mel scale: humans perceive pitch logarithmically.  These two formulas
// implement the most common "HTK" mel scale.
template <typename T>
inline T hz_to_mel(T hz) {
    return T(2595) * std::log10(T(1) + hz / T(700));
}
template <typename T>
inline T mel_to_hz(T mel) {
    return T(700) * (std::pow(T(10), mel / T(2595)) - T(1));
}

// Triangular mel filterbank as a (n_mels x n_bins) matrix.
// Each filter has unit peak; we don't area-normalise.  The network can
// absorb any constant scaling, and downstream standardisation will
// remove per-filter offsets.
template <typename T>
std::vector<std::vector<T>>
mel_filterbank(std::size_t n_mels, std::size_t n_bins, T sample_rate,
               T fmin = T(0), T fmax = T(-1))
{
    if (fmax <= T(0)) fmax = sample_rate * T(0.5);

    // n_mels + 2 evenly-spaced points in mel space, converted to
    // fractional FFT bin indices.  The +2 lets each filter use a
    // left/center/right triplet of consecutive points.
    const T mel_min = hz_to_mel(fmin);
    const T mel_max = hz_to_mel(fmax);
    std::vector<T> bin_points(n_mels + 2);
    for (std::size_t i = 0; i < n_mels + 2; ++i) {
        const T mel = mel_min + (mel_max - mel_min) * T(i) / T(n_mels + 1);
        const T hz  = mel_to_hz(mel);
        bin_points[i] = hz * T(2 * (n_bins - 1)) / sample_rate;
    }

    std::vector<std::vector<T>> filters(n_mels, std::vector<T>(n_bins, T(0)));
    for (std::size_t m = 0; m < n_mels; ++m) {
        const T left   = bin_points[m];
        const T center = bin_points[m + 1];
        const T right  = bin_points[m + 2];
        for (std::size_t k = 0; k < n_bins; ++k) {
            const T x = T(k);
            if (x >= left && x <= center && center > left)
                filters[m][k] = (x - left) / (center - left);
            else if (x > center && x <= right && right > center)
                filters[m][k] = (right - x) / (right - center);
        }
    }
    return filters;
}

// DCT-II, keeping only the first `num_keep` coefficients:
//   X[k] = sum_n  x[n] * cos(pi * k * (n + 0.5) / N)
// We don't apply the orthonormal scaling factor; standardisation
// downstream removes any constant per-coefficient offset.
template <typename T>
std::vector<T> dct2(const std::vector<T>& x, std::size_t num_keep) {
    const std::size_t N = x.size();
    std::vector<T> out(num_keep);
    for (std::size_t k = 0; k < num_keep; ++k) {
        T sum = T(0);
        const T factor = pi_v<T> * T(k) / T(N);
        for (std::size_t n = 0; n < N; ++n)
            sum += x[n] * std::cos(factor * (T(n) + T(0.5)));
        out[k] = sum;
    }
    return out;
}

// Common framing/windowing/accumulation loop, parametrised by a per-frame
// feature function.  FrameFn takes a const ref to the windowed frame and
// returns a vector<T> of length `feature_size`.
template <typename T, typename FrameFn>
std::vector<T> energy_weighted_features(const std::vector<T>& samples,
                                         std::size_t frame_size,
                                         std::size_t hop_size,
                                         std::size_t feature_size,
                                         FrameFn fn)
{
    if (samples.size() < frame_size)
        throw std::runtime_error("AudioFeatures: input shorter than frame_size");

    auto window = hann_window<T>(frame_size);
    std::vector<T> feature(feature_size, T(0));
    std::vector<T> frame(frame_size);
    T total_energy = T(0);

    for (std::size_t start = 0;
         start + frame_size <= samples.size();
         start += hop_size)
    {
        T energy = T(0);
        for (std::size_t i = 0; i < frame_size; ++i) {
            frame[i] = samples[start + i] * window[i];
            energy  += frame[i] * frame[i];
        }
        if (energy < T(1e-12)) continue;   // skip silent frames

        const auto frame_feature = fn(frame);
        for (std::size_t i = 0; i < feature_size; ++i)
            feature[i] += energy * frame_feature[i];
        total_energy += energy;
    }
    if (total_energy > T(0))
        for (auto& v : feature) v /= total_energy;
    return feature;
}

} // namespace detail

// ---- Public feature functions --------------------------------------------

// Per-frame log-magnitude spectra.  Returns one vector per non-silent
// frame; vector length is frame_size/2 + 1.  Used by the classifier (with
// per-file aggregation at eval time) and by future autoencoder work.
//
// Silent frames are skipped so we don't waste training capacity on noise
// in leading/trailing silence.
template <typename T>
std::vector<std::vector<T>>
logmag_spectrum_frames(const std::vector<T>& samples,
                       std::size_t frame_size = 4096,
                       std::size_t hop_size = 2048)
{
    if (samples.size() < frame_size)
        throw std::runtime_error("AudioFeatures: input shorter than frame_size");

    const std::size_t n_bins = frame_size / 2 + 1;
    auto window = detail::hann_window<T>(frame_size);
    std::vector<std::vector<T>> out;
    std::vector<T> frame(frame_size);

    for (std::size_t start = 0;
         start + frame_size <= samples.size();
         start += hop_size)
    {
        T energy = T(0);
        for (std::size_t i = 0; i < frame_size; ++i) {
            frame[i] = samples[start + i] * window[i];
            energy  += frame[i] * frame[i];
        }
        if (energy < T(1e-12)) continue;

        auto mag = detail::frame_magnitudes(frame);
        std::vector<T> log_mag(n_bins);
        for (std::size_t i = 0; i < n_bins; ++i)
            log_mag[i] = std::log(T(1) + mag[i]);
        out.push_back(std::move(log_mag));
    }
    return out;
}

// Per-frame MFCCs.  Same shape as logmag_spectrum_frames but with num_mfcc
// features per frame.
template <typename T>
std::vector<std::vector<T>>
mfcc_frames(const std::vector<T>& samples,
            T sample_rate,
            std::size_t frame_size = 4096,
            std::size_t hop_size = 2048,
            std::size_t num_mel_bins = 40,
            std::size_t num_mfcc = 13)
{
    if (samples.size() < frame_size)
        throw std::runtime_error("AudioFeatures: input shorter than frame_size");

    const std::size_t n_bins = frame_size / 2 + 1;
    auto window     = detail::hann_window<T>(frame_size);
    auto filterbank = detail::mel_filterbank<T>(num_mel_bins, n_bins, sample_rate);
    std::vector<std::vector<T>> out;
    std::vector<T> frame(frame_size);

    for (std::size_t start = 0;
         start + frame_size <= samples.size();
         start += hop_size)
    {
        T energy = T(0);
        for (std::size_t i = 0; i < frame_size; ++i) {
            frame[i] = samples[start + i] * window[i];
            energy  += frame[i] * frame[i];
        }
        if (energy < T(1e-12)) continue;

        auto mag = detail::frame_magnitudes(frame);
        std::vector<T> power(n_bins);
        for (std::size_t i = 0; i < n_bins; ++i)
            power[i] = mag[i] * mag[i];

        std::vector<T> log_mel(num_mel_bins);
        for (std::size_t m = 0; m < num_mel_bins; ++m) {
            T sum = T(0);
            for (std::size_t k = 0; k < n_bins; ++k)
                sum += filterbank[m][k] * power[k];
            log_mel[m] = std::log(sum + T(1e-12));
        }
        out.push_back(detail::dct2(log_mel, num_mfcc));
    }
    return out;
}

// Energy-weighted log-magnitude spectrum.
// Returns a vector of length frame_size/2 + 1.
template <typename T>
std::vector<T> logmag_spectrum(const std::vector<T>& samples,
                                std::size_t frame_size = 4096,
                                std::size_t hop_size = 2048)
{
    const std::size_t n_bins = frame_size / 2 + 1;
    return detail::energy_weighted_features<T>(
        samples, frame_size, hop_size, n_bins,
        [n_bins](const std::vector<T>& frame) {
            auto mag = detail::frame_magnitudes(frame);
            std::vector<T> log_mag(n_bins);
            for (std::size_t i = 0; i < n_bins; ++i)
                log_mag[i] = std::log(T(1) + mag[i]);
            return log_mag;
        });
}

// Energy-weighted average of frame-wise MFCCs.
// Returns a vector of length num_mfcc.
template <typename T>
std::vector<T> mfcc(const std::vector<T>& samples,
                    T sample_rate,
                    std::size_t frame_size = 4096,
                    std::size_t hop_size = 2048,
                    std::size_t num_mel_bins = 40,
                    std::size_t num_mfcc = 13)
{
    const std::size_t n_bins = frame_size / 2 + 1;
    auto filterbank = detail::mel_filterbank<T>(num_mel_bins, n_bins, sample_rate);

    return detail::energy_weighted_features<T>(
        samples, frame_size, hop_size, num_mfcc,
        [&, n_bins, num_mel_bins, num_mfcc](const std::vector<T>& frame) {
            auto mag = detail::frame_magnitudes(frame);

            // Power spectrum.
            std::vector<T> power(n_bins);
            for (std::size_t i = 0; i < n_bins; ++i)
                power[i] = mag[i] * mag[i];

            // Apply each mel filter to the power spectrum, then log.
            std::vector<T> log_mel(num_mel_bins);
            for (std::size_t m = 0; m < num_mel_bins; ++m) {
                T sum = T(0);
                for (std::size_t k = 0; k < n_bins; ++k)
                    sum += filterbank[m][k] * power[k];
                log_mel[m] = std::log(sum + T(1e-12));
            }

            // DCT-II of the log-mel-spectrum, keep first num_mfcc.
            return detail::dct2(log_mel, num_mfcc);
        });
}

} // namespace weft
