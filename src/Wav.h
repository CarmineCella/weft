#pragma once
//
// Wav.h -- read 16-bit PCM RIFF/WAV files into a vector of floats.
//
// The subset of the format we handle:
//   - PCM linear coding   (format code 1)
//   - 16 bits per sample, signed little-endian
//   - mono, or stereo downmixed to mono by averaging channels
//   - any sample rate; returned alongside the samples
//
// Deliberately unsupported (will throw):
//   - float WAVs        (format code 3)
//   - WAVEFORMATEXTENSIBLE (format code 0xFFFE)
//   - 8/24/32-bit PCM
//   - compressed formats (ADPCM, mu-law, etc.)
//
// We walk all chunks rather than assuming "fmt " then "data" in order,
// because real-world WAVs often have metadata chunks (LIST, INFO, ...)
// interleaved.  Unknown chunks are skipped over.
//
// Endianness:  we assume a little-endian host (x86, ARM Apple Silicon,
// 99% of what anyone runs in 2026).  All multi-byte values are read by
// hand from raw bytes so the parse itself is host-endian-agnostic; the
// only place we lean on host endianness is treating int16_t as a value
// after we've assembled the bytes -- and we use the explicit cast from
// uint16_t to int16_t which is well-defined for the bit pattern.
//
#include "Endian.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

struct WavData {
    std::vector<float> samples;        // mono, normalised to [-1, 1]
    std::uint32_t      sample_rate;
};

namespace detail {

inline std::string read_fourcc(std::ifstream& in) {
    char fourcc[4];
    in.read(fourcc, 4);
    if (!in) throw std::runtime_error("WAV: unexpected end of file");
    return std::string(fourcc, 4);
}

} // namespace detail

inline WavData load_wav(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("WAV: cannot open " + path);

    // RIFF / WAVE header
    if (detail::read_fourcc(in) != "RIFF")
        throw std::runtime_error("WAV: missing RIFF header: " + path);
    detail::read_le_u32(in);                       // file size - 8, ignored
    if (detail::read_fourcc(in) != "WAVE")
        throw std::runtime_error("WAV: not a WAVE file: " + path);

    // Walk chunks until we've seen both fmt and data.
    std::uint16_t format_code     = 0;
    std::uint16_t channels        = 0;
    std::uint32_t sample_rate     = 0;
    std::uint16_t bits_per_sample = 0;
    std::vector<unsigned char> raw_data;
    bool got_fmt  = false;
    bool got_data = false;

    while (!(got_fmt && got_data)) {
        if (!in || in.peek() == EOF) break;

        std::string   chunk_id   = detail::read_fourcc(in);
        std::uint32_t chunk_size = detail::read_le_u32(in);

        if (chunk_id == "fmt ") {
            format_code     = detail::read_le_u16(in);
            channels        = detail::read_le_u16(in);
            sample_rate     = detail::read_le_u32(in);
            detail::read_le_u32(in);               // byte rate, ignored
            detail::read_le_u16(in);               // block align, ignored
            bits_per_sample = detail::read_le_u16(in);
            // Skip any fmt extension bytes (chunk_size > 16 means there are some).
            if (chunk_size > 16)
                in.seekg(chunk_size - 16, std::ios::cur);
            got_fmt = true;
        }
        else if (chunk_id == "data") {
            if (!got_fmt)
                throw std::runtime_error("WAV: data chunk before fmt: " + path);
            raw_data.resize(chunk_size);
            in.read(reinterpret_cast<char*>(raw_data.data()), chunk_size);
            if (!in)
                throw std::runtime_error("WAV: short read in data chunk: " + path);
            // RIFF chunks are word-aligned; skip a pad byte if the size is odd.
            if (chunk_size & 1u) in.seekg(1, std::ios::cur);
            got_data = true;
        }
        else {
            // Unknown chunk: skip its body.
            in.seekg(chunk_size, std::ios::cur);
            if (chunk_size & 1u) in.seekg(1, std::ios::cur);
        }
    }

    if (!got_fmt || !got_data)
        throw std::runtime_error("WAV: missing fmt or data chunk: " + path);
    if (format_code != 1)
        throw std::runtime_error("WAV: not PCM (format code " +
                                 std::to_string(format_code) + "): " + path);
    if (bits_per_sample != 16)
        throw std::runtime_error("WAV: not 16-bit (" +
                                 std::to_string(bits_per_sample) + "): " + path);
    if (channels != 1 && channels != 2)
        throw std::runtime_error("WAV: unsupported channel count (" +
                                 std::to_string(channels) + "): " + path);

    // Reassemble int16 samples byte-by-byte (endian-safe).
    const std::size_t n_samples = raw_data.size() / 2;
    const std::size_t n_frames  = n_samples / channels;
    constexpr float   scale     = 1.0f / 32768.0f;

    WavData out;
    out.sample_rate = sample_rate;
    out.samples.resize(n_frames);

    auto sample_at = [&](std::size_t idx) -> std::int16_t {
        std::uint16_t u = std::uint16_t(raw_data[2 * idx])
                        | (std::uint16_t(raw_data[2 * idx + 1]) << 8);
        return static_cast<std::int16_t>(u);
    };

    if (channels == 1) {
        for (std::size_t i = 0; i < n_frames; ++i)
            out.samples[i] = sample_at(i) * scale;
    } else { // channels == 2: average L and R
        for (std::size_t i = 0; i < n_frames; ++i) {
            float L = sample_at(2 * i)     * scale;
            float R = sample_at(2 * i + 1) * scale;
            out.samples[i] = 0.5f * (L + R);
        }
    }
    return out;
}

// Write mono 16-bit PCM WAV.  `samples` are floats in [-1, 1]; values
// outside the range are clamped.  Mirrors load_wav -- save then load
// round-trips to within 16-bit quantisation.
//
// Layout (the minimal canonical WAV):
//     "RIFF" <u32 file_size-8> "WAVE"
//     "fmt " <u32 16> <u16 1=PCM> <u16 channels=1> <u32 sample_rate>
//            <u32 byte_rate> <u16 block_align> <u16 bits=16>
//     "data" <u32 data_bytes> <int16 samples...>
//
inline void save_wav(const std::string& path,
                     const std::vector<float>& samples,
                     std::uint32_t sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("WAV: cannot open for write: " + path);

    const std::uint16_t channels    = 1;
    const std::uint16_t bits        = 16;
    const std::uint16_t block_align = channels * bits / 8;
    const std::uint32_t byte_rate   = sample_rate * block_align;
    const std::uint32_t data_bytes  =
        static_cast<std::uint32_t>(samples.size()) * block_align;

    out.write("RIFF", 4);
    detail::write_le_u32(out, 36 + data_bytes);     // file size - 8
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    detail::write_le_u32(out, 16);                  // fmt chunk size
    detail::write_le_u16(out, 1);                   // PCM
    detail::write_le_u16(out, channels);
    detail::write_le_u32(out, sample_rate);
    detail::write_le_u32(out, byte_rate);
    detail::write_le_u16(out, block_align);
    detail::write_le_u16(out, bits);

    out.write("data", 4);
    detail::write_le_u32(out, data_bytes);
    for (float s : samples) {
        s = std::max(-1.0f, std::min(1.0f, s));
        // Symmetric mapping; 32767 keeps +1.0 and -1.0 in range.
        auto v = static_cast<std::int16_t>(std::lround(s * 32767.0f));
        detail::write_le_u16(out, static_cast<std::uint16_t>(v));
    }

    if (!out) throw std::runtime_error("WAV: write failed: " + path);
}

// Convenience overload taking a WavData.
inline void save_wav(const std::string& path, const WavData& wav) {
    save_wav(path, wav.samples, wav.sample_rate);
}

} // namespace weft
