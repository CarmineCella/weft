// test_audio_features.cpp
//
// Tests for hann_window, mel_filterbank, dct2, logmag_spectrum, mfcc.
//
#include "AudioFeatures.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using weft::logmag_spectrum;
using weft::mfcc;
using weft::detail::hann_window;
using weft::detail::mel_filterbank;
using weft::detail::dct2;
using weft::detail::hz_to_mel;
using weft::detail::mel_to_hz;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

int main() {
    std::cout << "weft :: AudioFeatures tests\n";

    constexpr float PI = 3.14159265358979323846f;

    // ---- 1. Hann window ----
    {
        auto w = hann_window<float>(8);
        check(w.size() == 8,                      "hann: correct length");
        check(close(w[0], 0.0f) && close(w[7], 0.0f),
              "hann: zero at endpoints");
        // The maximum of a Hann window of length N is at index (N-1)/2;
        // for N=8 that's between index 3 and 4, both equal to ~0.97.
        check(w[3] > 0.9f && w[4] > 0.9f,        "hann: peaks near middle");
    }

    // ---- 2. Hann window of length 1 is the identity ----
    {
        auto w = hann_window<float>(1);
        check(w.size() == 1 && close(w[0], 1.0f),
              "hann: length 1 is [1]");
    }

    // ---- 3. Mel scale round trip ----
    {
        float hz = 440.0f;
        float roundtrip = mel_to_hz(hz_to_mel(hz));
        check(close(roundtrip, hz, 0.01f),
              "mel: hz_to_mel and mel_to_hz are inverses");

        // Mel at 0 Hz should be 0
        check(close(hz_to_mel(0.0f), 0.0f),     "mel: 0 Hz -> 0 mel");
        // Mel scale is monotonically increasing
        check(hz_to_mel(1000.0f) < hz_to_mel(2000.0f),
              "mel: monotonic");
    }

    // ---- 4. Mel filterbank ----
    {
        const std::size_t n_mels = 10;
        const std::size_t n_bins = 257;     // half of 512 + 1
        const float       sr     = 44100.0f;
        auto fb = mel_filterbank<float>(n_mels, n_bins, sr);

        check(fb.size() == n_mels,           "mel filterbank: correct number of filters");
        check(fb[0].size() == n_bins,        "mel filterbank: each filter has n_bins entries");

        // All values are non-negative
        bool nn = true;
        for (auto& f : fb)
            for (auto v : f)
                if (v < 0) nn = false;
        check(nn,                             "mel filterbank: all values >= 0");

        // Each filter has at least one non-zero entry
        bool all_nonzero = true;
        for (auto& f : fb) {
            float s = 0;
            for (auto v : f) s += v;
            if (s <= 0) all_nonzero = false;
        }
        check(all_nonzero,                    "mel filterbank: each filter has some support");

        // Low filters cover lower frequencies than high filters.  We
        // check this by finding each filter's centre-of-mass bin.
        auto centre_of_mass = [&](const std::vector<float>& f) {
            float num = 0, den = 0;
            for (std::size_t k = 0; k < f.size(); ++k) {
                num += k * f[k];
                den += f[k];
            }
            return num / den;
        };
        check(centre_of_mass(fb[0]) < centre_of_mass(fb[n_mels - 1]),
              "mel filterbank: low filters centred below high filters");
    }

    // ---- 5. DCT-II ----
    {
        // Constant input: only coefficient 0 is non-zero, equal to N * c.
        std::vector<float> x(8, 3.0f);
        auto X = dct2(x, 8);
        bool ok = close(X[0], 8.0f * 3.0f);
        for (std::size_t i = 1; i < 8; ++i)
            if (!close(X[i], 0.0f)) ok = false;
        check(ok, "dct2: constant input has only DC coefficient");

        // Linearity: dct2(a*x + b*y) == a*dct2(x) + b*dct2(y)
        std::vector<float> y(8);
        for (std::size_t i = 0; i < 8; ++i) y[i] = std::cos(0.7f * i);
        std::vector<float> z(8);
        const float a = 1.5f, b = -2.0f;
        for (std::size_t i = 0; i < 8; ++i) z[i] = a * x[i] + b * y[i];
        auto Xt = dct2(x, 8);
        auto Yt = dct2(y, 8);
        auto Zt = dct2(z, 8);
        bool lin = true;
        for (std::size_t k = 0; k < 8; ++k)
            if (!close(Zt[k], a * Xt[k] + b * Yt[k])) lin = false;
        check(lin, "dct2: linearity");
    }

    // ---- 6. logmag_spectrum on a pure sine ----
    //   x[n] = sin(2 pi f n / sr) with f = sr * k / N for some bin k:
    //   the spectrum should peak strongly at bin k.
    {
        const std::size_t N    = 1024;
        const float       sr   = 44100.0f;
        const std::size_t k0   = 50;          // peak bin
        const float       freq = sr * k0 / N;
        const std::size_t n_samples = 8192;

        std::vector<float> samples(n_samples);
        for (std::size_t n = 0; n < n_samples; ++n)
            samples[n] = std::sin(2 * PI * freq * n / sr);

        auto feature = logmag_spectrum(samples, N, N / 2);
        check(feature.size() == N / 2 + 1,
              "logmag: output length = N/2 + 1");

        // Find the argmax bin
        std::size_t arg = 0;
        for (std::size_t i = 1; i < feature.size(); ++i)
            if (feature[i] > feature[arg]) arg = i;

        // Allow ±1 bin slack from windowing.
        check(std::abs(static_cast<int>(arg) - static_cast<int>(k0)) <= 1,
              "logmag: pure sine peaks at the expected bin");
    }

    // ---- 7. logmag_spectrum on a too-short input throws ----
    {
        std::vector<float> samples(100);   // shorter than default 4096
        bool threw = false;
        try { logmag_spectrum(samples); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "logmag: short input throws");
    }

    // ---- 8. MFCC: different signals give different features ----
    {
        const std::size_t N   = 1024;
        const float       sr  = 44100.0f;
        const std::size_t n   = 8192;

        std::vector<float> sine(n), saw(n);
        for (std::size_t i = 0; i < n; ++i) {
            sine[i] = std::sin(2 * PI * 440.0f * i / sr);
            // crude sawtooth -- many harmonics, very different timbre
            float t = float(i) / sr;
            float ph = std::fmod(t * 440.0f, 1.0f);
            saw[i] = 2.0f * ph - 1.0f;
        }

        auto m_sine = mfcc<float>(sine, sr, N, N / 2);
        auto m_saw  = mfcc<float>(saw,  sr, N, N / 2);

        check(m_sine.size() == 13, "mfcc: default length is 13");

        // Total absolute difference should be substantial.
        float diff = 0;
        for (std::size_t i = 0; i < m_sine.size(); ++i)
            diff += std::fabs(m_sine[i] - m_saw[i]);
        check(diff > 1.0f,
              "mfcc: sine and sawtooth produce different features");
    }

    // ---- 9. MFCC custom num_mfcc ----
    {
        std::vector<float> samples(8192);
        for (std::size_t i = 0; i < samples.size(); ++i)
            samples[i] = std::sin(0.1f * i);
        auto m = mfcc<float>(samples, 44100.0f, 1024, 512, 40, 20);
        check(m.size() == 20, "mfcc: respects num_mfcc parameter");
    }

    // ---- 10. logmag_spectrum_frames: multiple frames, correct shape ----
    {
        const std::size_t N    = 1024;
        const std::size_t n    = 8192;       // ~14 frames at hop N/2
        std::vector<float> samples(n);
        for (std::size_t i = 0; i < n; ++i)
            samples[i] = std::sin(0.1f * i);

        auto frames = weft::logmag_spectrum_frames(samples, N, N / 2);
        check(!frames.empty(),                       "logmag_frames: returns frames");
        check(frames[0].size() == N / 2 + 1,         "logmag_frames: each frame is N/2+1 long");

        // Number of non-silent frames should be close to (n - N) / hop + 1.
        // For sin(0.1*i) on 8192 samples with frame 1024 hop 512, all frames
        // have non-trivial energy.
        std::size_t expected = (n - N) / (N / 2) + 1;
        check(frames.size() == expected,
              "logmag_frames: count matches non-silent frame count");
    }

    // ---- 11. logmag_spectrum_frames: silent frames are skipped ----
    {
        const std::size_t N = 1024;
        std::vector<float> samples(8192, 0.0f);   // all zeros = all silent
        auto frames = weft::logmag_spectrum_frames(samples, N, N / 2);
        check(frames.empty(),
              "logmag_frames: silent input produces no frames");
    }

    // ---- 12. mfcc_frames: correct shape ----
    {
        const std::size_t N = 1024;
        std::vector<float> samples(8192);
        for (std::size_t i = 0; i < samples.size(); ++i)
            samples[i] = std::sin(0.2f * i);

        auto frames = weft::mfcc_frames<float>(samples, 44100.0f, N, N / 2);
        check(!frames.empty(),               "mfcc_frames: returns frames");
        check(frames[0].size() == 13,        "mfcc_frames: default 13 MFCCs per frame");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
