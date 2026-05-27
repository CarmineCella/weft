// extract_features / extract_features.cpp
//
// Scan a directory of WAV files, compute features for each one, and
// save the resulting (feature_matrix, labels, class_names, type) to a
// binary file that sound_classifier can read.
//
// Splitting feature extraction from classification means we pay the
// slow FFT-and-mel-filterbank cost once, then iterate quickly on
// architectures and hyperparameters with sound_classifier.
//
// Usage:
//     extract_features <data_dir> <output.feat> [logmag|mfcc]
//
// Filename convention is the same as before: the class label is the
// first dash-delimited token of the stem (e.g. "Bn-fp-B1-fp.wav" -> "Bn").
//
#include "AudioFeatures.h"
#include "FeatureCache.h"
#include "Matrix.h"
#include "Wav.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace weft;
namespace fs = std::filesystem;

static std::string parse_class(const fs::path& path) {
    const std::string stem = path.stem().string();
    const auto dash = stem.find('-');
    return (dash == std::string::npos) ? stem : stem.substr(0, dash);
}

static std::size_t feature_size(const std::string& type, std::size_t frame_size) {
    if (type == "logmag") return frame_size / 2 + 1;
    if (type == "mfcc")   return 13;
    throw std::runtime_error("unknown feature type: " + type);
}

static std::vector<float>
extract_feature(const std::vector<float>& samples, float sample_rate,
                const std::string& type,
                std::size_t frame_size, std::size_t hop_size)
{
    if (type == "logmag")
        return logmag_spectrum<float>(samples, frame_size, hop_size);
    return mfcc<float>(samples, sample_rate, frame_size, hop_size);
}

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;

    if (argc < 3) {
        std::cerr << "usage: extract_features <data_dir> <output.feat> [logmag|mfcc]\n";
        return 1;
    }
    const std::string data_dir     = argv[1];
    const std::string out_path     = argv[2];
    const std::string feature_type = (argc > 3) ? argv[3] : "mfcc";

    if (feature_type != "logmag" && feature_type != "mfcc") {
        std::cerr << "feature type must be 'logmag' or 'mfcc'\n";
        return 1;
    }

    const std::size_t FRAME_SIZE = 4096;
    const std::size_t HOP_SIZE   = 2048;
    const std::size_t FEAT_DIM   = feature_size(feature_type, FRAME_SIZE);

    std::cout << "weft :: extract_features\n";
    std::cout << "data dir:   " << data_dir << "\n";
    std::cout << "feature:    " << feature_type << " (" << FEAT_DIM << " dim)\n";
    std::cout << "output:     " << out_path << "\n\n";

    // ---- Scan files ----
    std::cout << "scanning..." << std::flush;
    std::vector<fs::path> wav_paths;
    try {
        for (const auto& entry : fs::directory_iterator(data_dir))
            if (entry.path().extension() == ".wav")
                wav_paths.push_back(entry.path());
    } catch (const std::exception& e) {
        std::cerr << "\nerror scanning " << data_dir << ": " << e.what() << "\n";
        return 1;
    }
    std::sort(wav_paths.begin(), wav_paths.end());
    std::cout << " " << wav_paths.size() << " WAV files\n";
    if (wav_paths.empty()) {
        std::cerr << "no .wav files found\n";
        return 1;
    }

    // ---- Extract ----
    std::cout << "extracting features " << std::flush;
    auto t_start = clock::now();

    std::map<std::string, int>      class_to_id;
    std::vector<std::vector<float>> features;
    std::vector<int>                labels;
    features.reserve(wav_paths.size());
    labels.reserve(wav_paths.size());

    std::size_t n_errors = 0;
    for (std::size_t i = 0; i < wav_paths.size(); ++i) {
        const auto& path = wav_paths[i];
        try {
            WavData wav = load_wav(path.string());
            auto feat = extract_feature(wav.samples,
                                         static_cast<float>(wav.sample_rate),
                                         feature_type, FRAME_SIZE, HOP_SIZE);
            if (feat.size() != FEAT_DIM)
                throw std::runtime_error("feature size mismatch");

            const std::string cls = parse_class(path);
            int id;
            auto it = class_to_id.find(cls);
            if (it != class_to_id.end()) {
                id = it->second;
            } else {
                id = static_cast<int>(class_to_id.size());
                class_to_id[cls] = id;
            }

            features.push_back(std::move(feat));
            labels.push_back(id);
        } catch (const std::exception& e) {
            ++n_errors;
            if (n_errors <= 5)
                std::cerr << "\n  warning: " << path.filename().string()
                          << ": " << e.what();
            if (n_errors == 5)
                std::cerr << "\n  (suppressing further warnings)";
        }
        if ((i + 1) % 100 == 0) std::cout << "." << std::flush;
    }
    double elapsed = std::chrono::duration<double>(clock::now() - t_start).count();
    std::cout << " done (" << std::fixed << std::setprecision(1)
              << elapsed << "s";
    if (n_errors > 0) std::cout << ", " << n_errors << " errors";
    std::cout << ")\n";

    const std::size_t N = features.size();
    const int         K = static_cast<int>(class_to_id.size());
    if (N == 0) {
        std::cerr << "no files processed successfully\n";
        return 1;
    }

    // ---- Build matrix + class-name vector + save ----
    Matrix<float> X(FEAT_DIM, N);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t k = 0; k < FEAT_DIM; ++k)
            X(k, j) = features[j][k];
    features.clear(); features.shrink_to_fit();

    std::vector<std::string> class_names(K);
    for (const auto& [name, id] : class_to_id)
        class_names[id] = name;

    CachedFeatures cache;
    cache.X            = std::move(X);
    cache.labels       = std::move(labels);
    cache.class_names  = std::move(class_names);
    cache.feature_type = feature_type;

    try {
        save_features(out_path, cache);
    } catch (const std::exception& e) {
        std::cerr << "save failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nsaved " << N << " samples, " << K << " classes to "
              << out_path << "\n";
    return 0;
}
