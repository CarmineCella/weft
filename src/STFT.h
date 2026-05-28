#pragma once
//
// STFT.h  --  short-time Fourier transform and its inverse.
//
// The STFT slices a signal into overlapping frames, windows each one, and
// FFTs it, producing a sequence of complex spectra (one per frame).  The
// inverse (iSTFT) FFTs each spectrum back to the time domain and stitches
// the frames together with overlap-add.
//
// This is the bridge between a waveform and the per-frame spectral domain
// the autoencoder works in: WAV -> STFT -> magnitudes -> (model) ->
// magnitudes -> iSTFT -> WAV.
//
// Reconstruction (WOLA).  We apply a Hann window on analysis AND on
// synthesis, overlap-add the synthesised frames, and divide by the
// overlap-added window-power (sum of squared synthesis windows).  This
// "weighted overlap-add" inverts exactly wherever frames fully overlap,
// for any hop that satisfies the COLA condition (Hann at 50% hop does),
// and degrades gracefully at the signal edges where coverage is partial.
//
// Frames hold the FULL complex spectrum (frame_size bins, not the
// half-spectrum), which keeps the inverse a plain ifft with no
// conjugate-symmetry bookkeeping.  For real input the upper half is the
// mirror of the lower; an application that edits magnitudes must keep
// that symmetry (see rebuild_hermitian below).
//
#include "FFT.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace weft {

namespace detail {

template <typename T>
inline std::vector<T> hann(std::size_t n) {
    std::vector<T> w(n);
    for (std::size_t i = 0; i < n; ++i)
        w[i] = T(0.5) * (T(1) - std::cos(T(2) * pi_v<T> * i / (n - 1)));
    return w;
}

} // namespace detail

// Forward STFT.  Returns one full-length complex spectrum per frame.
// frame_size must be a power of two (FFT requirement).
template <typename T = float>
std::vector<std::vector<std::complex<T>>>
stft(const std::vector<T>& signal,
     std::size_t frame_size = 4096,
     std::size_t hop_size   = 2048)
{
    if (!detail::is_power_of_two(frame_size))
        throw std::invalid_argument("stft: frame_size must be a power of two");
    if (hop_size == 0)
        throw std::invalid_argument("stft: hop_size must be > 0");

    const auto window = detail::hann<T>(frame_size);
    std::vector<std::vector<std::complex<T>>> frames;

    if (signal.size() < frame_size) return frames;

    for (std::size_t start = 0;
         start + frame_size <= signal.size();
         start += hop_size)
    {
        std::vector<std::complex<T>> spectrum(frame_size);
        for (std::size_t i = 0; i < frame_size; ++i)
            spectrum[i] = std::complex<T>(signal[start + i] * window[i], T(0));
        fft(spectrum);
        frames.push_back(std::move(spectrum));
    }
    return frames;
}

// Inverse STFT.  `out_length` is the desired output length (typically the
// original signal length); pass 0 to size it from the frame count.
template <typename T = float>
std::vector<T>
istft(const std::vector<std::vector<std::complex<T>>>& frames,
      std::size_t frame_size = 4096,
      std::size_t hop_size   = 2048,
      std::size_t out_length = 0)
{
    if (frames.empty()) return {};

    const std::size_t n_frames = frames.size();
    const std::size_t needed   = (n_frames - 1) * hop_size + frame_size;
    const std::size_t len      = (out_length == 0) ? needed : out_length;

    const auto window = detail::hann<T>(frame_size);

    std::vector<T> out(len, T(0));
    std::vector<T> norm(len, T(0));   // accumulated synthesis window power

    for (std::size_t f = 0; f < n_frames; ++f) {
        std::vector<std::complex<T>> time = frames[f];   // copy; ifft is in-place
        ifft(time);

        const std::size_t start = f * hop_size;
        for (std::size_t i = 0; i < frame_size; ++i) {
            const std::size_t idx = start + i;
            if (idx >= len) break;
            const T w = window[i];
            out[idx]  += time[i].real() * w;   // synthesis window
            norm[idx] += w * w;                // window power for normalisation
        }
    }

    for (std::size_t i = 0; i < len; ++i)
        if (norm[i] > T(1e-8)) out[i] /= norm[i];

    return out;
}

// Helpers for magnitude/phase round-tripping, used by the audio VAE apps.
//
// magnitude(frame) and phase(frame) pull the half-spectrum (the unique
// bins 0 .. frame_size/2) out of a full STFT frame.
template <typename T = float>
std::vector<T> magnitude(const std::vector<std::complex<T>>& frame) {
    const std::size_t bins = frame.size() / 2 + 1;
    std::vector<T> mag(bins);
    for (std::size_t i = 0; i < bins; ++i) mag[i] = std::abs(frame[i]);
    return mag;
}

template <typename T = float>
std::vector<T> phase(const std::vector<std::complex<T>>& frame) {
    const std::size_t bins = frame.size() / 2 + 1;
    std::vector<T> ph(bins);
    for (std::size_t i = 0; i < bins; ++i) ph[i] = std::arg(frame[i]);
    return ph;
}

// Rebuild a full Hermitian-symmetric complex frame from a half-spectrum
// magnitude and phase, so ifft yields a real signal.  bins = frame_size/2+1
// values of magnitude/phase produce a frame_size-long complex frame with
// frame[frame_size-k] = conj(frame[k]).
template <typename T = float>
std::vector<std::complex<T>>
rebuild_hermitian(const std::vector<T>& mag, const std::vector<T>& ph) {
    if (mag.size() != ph.size())
        throw std::invalid_argument("rebuild_hermitian: size mismatch");
    const std::size_t bins = mag.size();          // frame_size/2 + 1
    const std::size_t N    = (bins - 1) * 2;       // frame_size
    std::vector<std::complex<T>> frame(N);
    for (std::size_t k = 0; k < bins; ++k)
        frame[k] = std::polar(mag[k], ph[k]);
    for (std::size_t k = 1; k < bins - 1; ++k)
        frame[N - k] = std::conj(frame[k]);
    return frame;
}

} // namespace weft
