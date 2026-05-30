#pragma once
//
// MelTransform.h  --  forward + inverse mel-frequency transforms for
//                     log-magnitude spectrograms.
//
// Why this exists.  ConvVAEs on audio need a frequency axis that's
// short enough to fit through small conv layers but expressive enough
// that harmonic structure survives the round trip.  Linear block-
// averaging (e.g. 32 raw bins -> 1 output bin) destroys harmonic peaks
// by smearing them into broad plateaus, which makes the reconstructed
// audio sound like noise even when the overall envelope is right.
// The mel scale concentrates frequency resolution where music actually
// lives (logarithmic spacing, dense in the low/mid kHz range), so the
// same 64 output bins capture many more harmonic peaks of a typical
// instrument tone than 64 linear bins would.
//
// Forward (linear log-mag -> log-mel-mag):
//   lin_mag    = expm1(lin_logmag)
//   mel_mag    = M * lin_mag                   (triangular filterbank)
//   mel_logmag = log1p(mel_mag)
// This is the standard "log-mel-spectrogram" recipe used everywhere
// in audio CNN work.
//
// Inverse (log-mel-mag -> linear magnitude for iSTFT):
//   mel_mag = expm1(mel_logmag)
//   lin_mag = M^T * mel_mag                    (lossy pseudo-inverse)
// The transpose isn't a true inverse of M -- the filterbanks overlap
// so M^T M is not identity -- but it's the standard "good enough"
// reconstruction, equivalent to spreading each mel bin's energy back
// over the triangle that originally fed it.  Quality is "blurry but
// faithful" -- comparable to mel-spec inversion in librosa.
//
// Construct one MelTransform up front and call its two methods on each
// frame.  Per-frame cost is O(n_mels * n_bins), about 130k floating
// adds for our default (n_mels=64, n_bins=2049).
//
#include "AudioFeatures.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace weft {

template <typename T = float>
class MelTransform {
public:
    // n_mels      number of mel bins (== output dim of the forward)
    // n_bins      number of linear bins per STFT frame (== FFT_size/2 + 1)
    // sample_rate Hz, used to map mel-spaced points to linear bin indices
    //
    // scale       a constant applied AFTER the log1p in the forward and
    //             undone BEFORE the expm1 in the inverse.  Mel binning
    //             concentrates energy compared to linear binning, so the
    //             raw log-mel-mag values can be ~5-10x larger than the
    //             original log-mag values for the same audio.  That
    //             pushes the VAE's randomly-initialised dense layers
    //             into a regime where KL explodes (logvar gets large,
    //             exp(logvar) is astronomical).  Dividing by `scale`
    //             brings values back into the 0-1 range the dense VAE
    //             was tuned for and stabilises training.  scale=8 works
    //             well in practice for typical music audio; the choice
    //             must be the same at train and inference time.
    MelTransform(std::size_t n_mels, std::size_t n_bins, T sample_rate,
                 T scale = T(8))
        : n_mels_(n_mels), n_bins_(n_bins), scale_(scale),
          M_(detail::mel_filterbank<T>(n_mels, n_bins, sample_rate)) {}

    // 2049-bin (or whatever n_bins_) log-magnitude column ->
    // 64-bin (or whatever n_mels_) log-mel-magnitude column, scaled.
    void linear_logmag_to_logmel(const T* lin_logmag, T* mel_logmag) const {
        for (std::size_t m = 0; m < n_mels_; ++m) {
            T lin_mel = T(0);
            for (std::size_t k = 0; k < n_bins_; ++k) {
                const T lm = std::max(T(0), lin_logmag[k]);
                lin_mel += M_[m][k] * std::expm1(lm);
            }
            mel_logmag[m] = std::log1p(std::max(T(0), lin_mel)) / scale_;
        }
    }

    // 64-bin scaled log-mel-magnitude column -> 2049-bin LINEAR magnitude
    // column.  Undoes the `scale_` division applied in the forward.
    void logmel_to_linear_mag(const T* mel_logmag, T* lin_mag) const {
        std::vector<T> mel_mag(n_mels_);
        for (std::size_t m = 0; m < n_mels_; ++m)
            mel_mag[m] = std::expm1(std::max(T(0), mel_logmag[m] * scale_));
        for (std::size_t k = 0; k < n_bins_; ++k) {
            T s = T(0);
            for (std::size_t m = 0; m < n_mels_; ++m)
                s += M_[m][k] * mel_mag[m];
            lin_mag[k] = std::max(T(0), s);
        }
    }

    std::size_t n_mels() const { return n_mels_; }
    std::size_t n_bins() const { return n_bins_; }
    T           scale()  const { return scale_; }

private:
    std::size_t n_mels_, n_bins_;
    T scale_;
    std::vector<std::vector<T>> M_;
};

} // namespace weft
