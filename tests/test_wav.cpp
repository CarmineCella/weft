// test_wav.cpp
//
// Tests for the WAV reader.  We write tiny synthetic WAV files to /tmp
// and read them back.  Verifies sample rate, length, channel mixdown,
// unknown-chunk skipping, and the standard error paths.
//
#include "Wav.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using weft::WavData;
using weft::load_wav;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

// ---- Byte-level WAV writer used only by tests ----------------------------

static void w16(std::ofstream& out, std::uint16_t x) {
    unsigned char b[2] = {
        static_cast<unsigned char>( x        & 0xff),
        static_cast<unsigned char>((x >> 8)  & 0xff)
    };
    out.write(reinterpret_cast<char*>(b), 2);
}
static void w32(std::ofstream& out, std::uint32_t x) {
    unsigned char b[4] = {
        static_cast<unsigned char>( x        & 0xff),
        static_cast<unsigned char>((x >>  8) & 0xff),
        static_cast<unsigned char>((x >> 16) & 0xff),
        static_cast<unsigned char>((x >> 24) & 0xff)
    };
    out.write(reinterpret_cast<char*>(b), 4);
}
static void w_sample(std::ofstream& out, std::int16_t s) {
    w16(out, static_cast<std::uint16_t>(s));
}

// Write a minimal valid WAV.  Optionally inject a "LIST" chunk between
// fmt and data so we can test unknown-chunk skipping.
static void write_wav(const std::string& path,
                      const std::vector<std::int16_t>& samples,
                      std::uint32_t sample_rate,
                      std::uint16_t channels,
                      bool with_extra_chunk = false)
{
    std::ofstream out(path, std::ios::binary);
    const std::uint32_t fmt_size   = 16;
    const std::uint32_t data_size  = static_cast<std::uint32_t>(samples.size() * 2);
    const std::uint32_t extra_size = with_extra_chunk ? 8 : 0;  // 8-byte LIST body

    // RIFF header
    out.write("RIFF", 4);
    w32(out, 4 + (8 + fmt_size) + (with_extra_chunk ? (8 + extra_size) : 0)
              + (8 + data_size));
    out.write("WAVE", 4);

    // fmt chunk
    out.write("fmt ", 4);
    w32(out, fmt_size);
    w16(out, 1);                                      // format = PCM
    w16(out, channels);
    w32(out, sample_rate);
    w32(out, sample_rate * channels * 2);             // byte rate
    w16(out, static_cast<std::uint16_t>(channels * 2)); // block align
    w16(out, 16);                                     // bits per sample

    // optional unknown chunk (LIST with 8 dummy bytes)
    if (with_extra_chunk) {
        out.write("LIST", 4);
        w32(out, extra_size);
        for (std::uint32_t i = 0; i < extra_size; ++i)
            out.put(static_cast<char>(0x42));
    }

    // data chunk
    out.write("data", 4);
    w32(out, data_size);
    for (auto s : samples) w_sample(out, s);
}

int main() {
    std::cout << "weft :: WAV tests\n";

    const std::string path = "/tmp/weft_test.wav";

    // ---- 1. Mono round trip ----
    {
        std::vector<std::int16_t> raw = {0, 16384, -16384, 32767, -32768};
        write_wav(path, raw, 44100, 1);

        WavData w = load_wav(path);
        std::remove(path.c_str());

        check(w.sample_rate == 44100, "mono: sample rate read correctly");
        check(w.samples.size() == raw.size(), "mono: sample count matches");
        bool values_ok =
            close(w.samples[0],  0.0f) &&
            close(w.samples[1],  16384.0f / 32768.0f) &&
            close(w.samples[2], -16384.0f / 32768.0f) &&
            close(w.samples[3],  32767.0f / 32768.0f) &&
            close(w.samples[4], -1.0f);
        check(values_ok, "mono: sample values normalised to [-1, 1]");
    }

    // ---- 2. Stereo is mixed down to mono ----
    //   L = +s, R = -s  ->  mono = 0
    //   L = +1, R = +1  ->  mono = +1
    {
        std::vector<std::int16_t> raw;
        raw.push_back( 16384); raw.push_back(-16384);   // frame 0: L+R -> 0
        raw.push_back( 32767); raw.push_back( 32767);   // frame 1: L=R -> ~+1
        raw.push_back(-32768); raw.push_back(-32768);   // frame 2: L=R -> -1
        write_wav(path, raw, 44100, 2);

        WavData w = load_wav(path);
        std::remove(path.c_str());

        check(w.samples.size() == 3, "stereo: 3 frames after mixdown");
        check(close(w.samples[0], 0.0f),               "stereo: L=-R averages to 0");
        check(close(w.samples[1], 32767.0f / 32768.0f), "stereo: L=R passes through");
        check(close(w.samples[2], -1.0f),              "stereo: L=R=-1 passes through");
    }

    // ---- 3. Unknown chunks are skipped ----
    {
        std::vector<std::int16_t> raw = {100, 200, 300};
        write_wav(path, raw, 22050, 1, /*with_extra_chunk=*/true);

        WavData w = load_wav(path);
        std::remove(path.c_str());

        check(w.sample_rate == 22050, "skips unknown chunks: sample rate ok");
        check(w.samples.size() == 3,  "skips unknown chunks: samples ok");
    }

    // ---- 4. Missing file throws ----
    {
        bool threw = false;
        try { load_wav("/tmp/weft_no_such_wav.wav"); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "missing file throws runtime_error");
    }

    // ---- 5. Non-RIFF garbage throws ----
    {
        std::ofstream out(path, std::ios::binary);
        out.write("NOPE", 4);
        w32(out, 0);
        out.write("WAVE", 4);
        out.close();

        bool threw = false;
        try { load_wav(path); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(path.c_str());

        check(threw, "non-RIFF file throws runtime_error");
    }

    // ---- 6. Non-PCM format throws ----
    {
        std::ofstream out(path, std::ios::binary);
        out.write("RIFF", 4); w32(out, 36);
        out.write("WAVE", 4);
        out.write("fmt ", 4); w32(out, 16);
        w16(out, 3);            // float format -- not supported
        w16(out, 1);            // channels
        w32(out, 44100);        // rate
        w32(out, 44100 * 4);    // byte rate
        w16(out, 4);            // block align
        w16(out, 32);           // bits per sample
        out.write("data", 4); w32(out, 0);
        out.close();

        bool threw = false;
        try { load_wav(path); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(path.c_str());

        check(threw, "non-PCM format throws runtime_error");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
