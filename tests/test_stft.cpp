// test_stft.cpp
//
// Tests for the short-time Fourier transform and its inverse.  The key
// property is that iSTFT(STFT(x)) reproduces x in the interior (away from
// the edges, where window coverage is partial).
//
#include "STFT.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

using weft::stft;
using weft::istft;
using weft::magnitude;
using weft::phase;
using weft::rebuild_hermitian;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}

int main() {
    std::cout << "weft :: STFT tests\n";

    const std::size_t N = 4096, H = 2048;

    // A test signal: sum of a few sinusoids, long enough for many frames.
    const std::size_t len = 16384;
    std::vector<float> sig(len);
    for (std::size_t i = 0; i < len; ++i)
        sig[i] = 0.5f * std::sin(2.0f * 3.14159265f * 0.01f * i)
               + 0.3f * std::sin(2.0f * 3.14159265f * 0.013f * i)
               + 0.2f * std::sin(2.0f * 3.14159265f * 0.027f * i);

    // ---- Frame count and bin count ----
    {
        auto frames = stft(sig, N, H);
        const std::size_t expected = (len - N) / H + 1;
        check(frames.size() == expected,        "stft: expected frame count");
        check(!frames.empty() && frames[0].size() == N,
              "stft: each frame holds full complex spectrum");
    }

    // ---- Round-trip reproduces the interior of the signal ----
    {
        auto frames = stft(sig, N, H);
        auto rec    = istft(frames, N, H, len);

        check(rec.size() == len, "istft: output length matches");

        // Compare the interior (skip the first and last frame, where the
        // overlap-add window coverage is incomplete).
        float max_err = 0;
        for (std::size_t i = N; i + N < len; ++i)
            max_err = std::max(max_err, std::fabs(rec[i] - sig[i]));
        check(max_err < 1e-4f, "round-trip: interior reproduced (err < 1e-4)");
    }

    // ---- Magnitude/phase extraction sizes ----
    {
        auto frames = stft(sig, N, H);
        auto mag = magnitude(frames[2]);
        auto ph  = phase(frames[2]);
        check(mag.size() == N / 2 + 1, "magnitude: half-spectrum length");
        check(ph.size()  == N / 2 + 1, "phase: half-spectrum length");
    }

    // ---- rebuild_hermitian inverts magnitude/phase for a real frame ----
    {
        auto frames = stft(sig, N, H);
        const auto& frame = frames[3];
        auto mag = magnitude(frame);
        auto ph  = phase(frame);
        auto rebuilt = rebuild_hermitian(mag, ph);

        check(rebuilt.size() == N, "rebuild_hermitian: full frame length");

        // Round-tripping through abs/arg/polar in float loses a few ulps,
        // and FFT magnitudes here run into the hundreds, so compare with a
        // RELATIVE tolerance against the frame's largest component.
        float max_err = 0, max_mag = 0;
        for (std::size_t k = 0; k < N; ++k) {
            max_err = std::max(max_err, std::fabs(rebuilt[k].real() - frame[k].real()));
            max_err = std::max(max_err, std::fabs(rebuilt[k].imag() - frame[k].imag()));
            max_mag = std::max(max_mag, std::abs(frame[k]));
        }
        check(max_err < 1e-4f * max_mag,
              "rebuild_hermitian: reproduces original complex frame (relative)");
    }

    // ---- Modified-magnitude path stays real after ifft ----
    //   Rebuild frames from (magnitude, phase) and confirm the resynthesis
    //   is still a faithful round-trip -- this is exactly the path the
    //   audio VAE apps use (edit magnitude, keep phase, rebuild, istft).
    {
        auto frames = stft(sig, N, H);
        std::vector<std::vector<std::complex<float>>> rebuilt;
        for (const auto& fr : frames)
            rebuilt.push_back(rebuild_hermitian(magnitude(fr), phase(fr)));
        auto rec = istft(rebuilt, N, H, len);

        float max_err = 0;
        for (std::size_t i = N; i + N < len; ++i)
            max_err = std::max(max_err, std::fabs(rec[i] - sig[i]));
        check(max_err < 1e-4f, "mag/phase rebuild path round-trips (err < 1e-4)");
    }

    // ---- Short signal yields no frames ----
    {
        std::vector<float> tiny(100, 0.1f);
        auto frames = stft(tiny, N, H);
        check(frames.empty(), "stft: signal shorter than a frame -> no frames");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
