#include "Conv2D.h"
#include "ConvNetwork.h"
#include "Dense.h"
#include "Dropout.h"
#include "LayerNorm.h"
#include "Network.h"
#include "ReLU.h"
#include "ReLU4D.h"
#include "Sigmoid.h"
#include "TokenTransformer.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
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
static std::vector<char> read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
static void write(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
int main() {
    using namespace weft;
    const auto dir = std::filesystem::temp_directory_path() /
        ("weft-checkpoint-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    const std::string path = (dir / "weights").string();
    Network<float> source, restored;
    auto& layer = source.add<Dense>(2, 2, 1);
    source.add<ReLU>();
    restored.add<Dense>(2, 2, 2); restored.add<ReLU>();
    source.save(path);
    restored.load(path);
    const auto valid = read(path);
    auto bytes = valid; bytes.resize(bytes.size() - 4); write(path, bytes);
    check(throws([&] { restored.load(path); }), "truncated dense checkpoint is rejected");
    bytes = valid;
    const std::uint32_t unsupported = 999;
    std::memcpy(bytes.data() + 4, &unsupported, sizeof(unsupported)); write(path, bytes);
    check(throws([&] { restored.load(path); }), "unsupported version is rejected");
    write(path, valid);
    Network<float> wrong_activation;
    wrong_activation.add<Dense>(2, 2); wrong_activation.add<Sigmoid>();
    check(throws([&] { wrong_activation.load(path); }), "same-shaped activation mismatch is rejected");
    Network<double> wrong_scalar;
    wrong_scalar.add<Dense>(2, 2); wrong_scalar.add<ReLU>();
    check(throws([&] { wrong_scalar.load(path); }), "scalar-width mismatch is rejected");
    source.save(path, "sample_rate=44100");
    check(throws([&] { restored.load(path, "sample_rate=48000"); }), "preprocessing mismatch is rejected");
    restored.load(path, "sample_rate=44100");
    check(restored.summary() == source.summary(), "matching metadata round-trips");
    Network<float> da, db;
    da.add<Dropout>(0.3001f); db.add<Dropout>(0.3002f);
    da.save(path);
    check(throws([&] { db.load(path); }), "dropout settings use full precision");
    Network<float> norm_a, norm_b;
    norm_a.add<LayerNorm>(2, 1e-4f); norm_b.add<LayerNorm>(2, 2e-4f);
    norm_a.save(path);
    check(throws([&] { norm_b.load(path); }), "normalization epsilon mismatch is rejected");
    {
        std::ofstream f(path, std::ios::binary);
        const std::uint32_t version = 1, count = 2;
        f.write("WNET", 4);
        f.write(reinterpret_cast<const char*>(&version), 4);
        f.write(reinterpret_cast<const char*>(&count), 4);
        layer.save_params(f);
    }
    check(!throws([&] { restored.load(path); }), "legacy version-one dense checkpoint remains readable");
    check(throws([&] { restored.load(path, "must-validate-metadata"); }),
          "legacy files cannot satisfy a metadata contract");
    ConvNetwork<float> conv;
    conv.add<Conv2D>(1, 2, 3); conv.add<ReLU4D>();
    conv.save(path);
    bytes = read(path);
    std::memcpy(bytes.data() + 4, &unsupported, 4); write(path, bytes);
    check(throws([&] { conv.load(path); }), "convolution checkpoint version is checked");
    write(path, {'W'});
    check(throws([&] { conv.load(path); }), "short convolution header is rejected");
    TokenTransformer<float> model(8, 4, 2, 1, 8, 8), copy(8, 4, 2, 1, 8, 8);
    model.save(path);
    const auto token_valid = read(path);
    check(!throws([&] { copy.load(path); }), "transformer header round-trips");
    bytes = token_valid; bytes.resize(bytes.size() - 4); write(path, bytes);
    check(throws([&] { copy.load(path); }), "truncated transformer checkpoint is rejected");
    write(path, token_valid);
    TokenTransformer<float> heads_mismatch(8, 4, 1, 1, 8, 8);
    check(throws([&] { heads_mismatch.load(path); }), "same-shaped attention-head mismatch is rejected");
    std::uint32_t config_length = 0;
    std::memcpy(&config_length, token_valid.data() + 16, 4);
    bytes.assign(token_valid.begin() + 20 + config_length, token_valid.end());
    write(path, bytes);
    check(!throws([&] { copy.load(path); }), "legacy headerless transformer remains readable");
    bytes.resize(bytes.size() - 4); write(path, bytes);
    check(throws([&] { copy.load(path); }), "truncated legacy transformer is rejected");
    std::filesystem::remove_all(dir);
    return failed ? 1 : 0;
}
