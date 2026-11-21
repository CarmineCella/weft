#include "AudioFeatures.h"
#include "AudioModelConfig.h"
#include "FeatureCache.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static int failed = 0;
static void check(bool ok, const std::string& name) {
    std::cout << (ok ? "[ ok ] " : "[FAIL] ") << name << '\n';
    if (!ok) ++failed;
}
template <typename F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
int main() {
    using namespace weft;
    std::vector<double> samples(32, 0.0);
    for (std::size_t i = 0; i < samples.size(); ++i)
        if (i < 8 || i >= 16) samples[i] = std::sin(0.4 * i + 0.2);
    std::vector<std::uint64_t> offsets;
    const auto frames = logmag_spectrum_frames(samples, 8, 8, &offsets);
    check(frames.size() == 3 && offsets == std::vector<std::uint64_t>{0, 16, 24},
          "silence removal retains original sample offsets");
    std::vector<std::uint64_t> mfcc_offsets;
    mfcc_frames(samples, 44100.0, 8, 8, 4, 3, &mfcc_offsets);
    check(mfcc_offsets == offsets, "MFCC and logmag timing agree");
    CachedFeatures cache;
    cache.X = Matrix<float>(5, frames.size());
    for (std::size_t j = 0; j < frames.size(); ++j)
        for (std::size_t i = 0; i < frames[j].size(); ++i) cache.X(i, j) = float(frames[j][i]);
    cache.labels = {0, 0, 0};
    cache.file_ids = {0, 0, 0};
    cache.class_names = {"tone"};
    cache.feature_type = "logmag";
    cache.per_frame = true;
    cache.sample_rate = 44100; cache.frame_size = 8; cache.hop_size = 8;
    cache.frame_offsets = offsets;
    const auto runs = contiguous_frame_runs(cache);
    check(runs == std::vector<std::pair<std::size_t, std::size_t>>{{0, 1}, {1, 3}},
          "temporal runs split at internal silence");
    cache.file_ids[2] = 1;
    check(contiguous_frame_runs(cache).size() == 3, "temporal runs also split at file boundaries");
    const std::string path = "/tmp/weft_audio_timing.feat";
    save_features(path, cache);
    const auto loaded = load_features(path);
    check(loaded.frame_offsets == offsets && loaded.sample_rate == 44100 &&
          loaded.frame_size == 8 && loaded.hop_size == 8 && loaded.window == "hann",
          "version-three cache round-trips timing and preprocessing");
    {
        std::ifstream f(path, std::ios::binary);
        std::vector<char> bytes{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
        f.close();
        bytes.resize(bytes.size() - 1);
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    check(throws([&] { load_features(path); }), "truncated timing payload is rejected");
    {
        // An independently written version-two fixture has no timing metadata.
        std::ofstream out(path, std::ios::binary);
        out.write("WFED", 4);
        for (std::uint32_t value : {2u, 1u, 1u, 1u, 1u})
            detail::write_le_u32(out, value);
        detail::write_string(out, "logmag");
        detail::write_string(out, "tone");
        const float value = 0.5f;
        const std::int32_t id = 0;
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
    const auto legacy = load_features(path);
    check(legacy.X(0, 0) == 0.5f && legacy.frame_offsets.empty(),
          "legacy version-two cache remains readable");
    check(throws([&] { contiguous_frame_runs(legacy); }),
          "legacy cache cannot silently imply temporal adjacency");
    check(throws([&] { AudioModelConfig::require_cache(legacy); }),
          "audio models require known preprocessing");
    check(throws([] { AudioModelConfig::require_sample_rate(48000); }),
          "wrong inference sample rate is rejected");
    check(throws([&] { logmag_spectrum_frames(samples, 8, 0); }),
          "zero hop is rejected instead of looping indefinitely");
    std::remove(path.c_str());
    return failed ? 1 : 0;
}
