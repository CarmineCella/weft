#pragma once
//
// Shared preprocessing contract for the audio VAE example programs.
// The exact string is stored alongside each model's architecture header.
//
#include "FeatureCache.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace weft {
struct AudioModelConfig {
    static constexpr std::uint32_t sample_rate = 44100;
    static constexpr std::size_t frame_size = 4096, hop_size = 2048;
    static constexpr std::size_t latent = 32, mel_bins = 64, time_frames = 16;
    static constexpr float mel_scale = 8.0f;

    static std::string metadata(bool convolutional = false) {
        std::string text = "logmag;sample_rate=" + std::to_string(sample_rate) +
            ";frame=" + std::to_string(frame_size) + ";hop=" + std::to_string(hop_size) +
            ";window=hann;latent=" + std::to_string(latent);
        if (convolutional)
            text += ";mels=" + std::to_string(mel_bins) +
                ";scale=" + std::to_string(mel_scale) +
                ";time=" + std::to_string(time_frames);
        return text;
    }
    static void require_sample_rate(std::uint32_t actual) {
        if (actual != sample_rate)
            throw std::runtime_error("audio VAE: expected 44100 Hz input; resample before use");
    }
    static void require_cache(const CachedFeatures& cache) {
        if (cache.sample_rate != sample_rate || cache.frame_size != frame_size ||
            cache.hop_size != hop_size || cache.window != "hann" ||
            cache.feature_type != "logmag" || !cache.per_frame ||
            cache.X.rows() != frame_size / 2 + 1 || cache.X.cols() == 0)
            throw std::runtime_error(
                "audio VAE: expected 44100 Hz, 4096/2048 Hann logmag frames; re-run extract_features");
    }
};
} // namespace weft
