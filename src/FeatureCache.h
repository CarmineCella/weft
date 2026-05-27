#pragma once
//
// FeatureCache.h -- save/load extracted feature datasets to a flat
//                   binary file, so the slow extraction step doesn't
//                   need to repeat on every classifier iteration.
//
// Usage:
//
//     CachedFeatures cache { X, labels, class_names, "mfcc" };
//     save_features("sol_mfcc.feat", cache);
//     // ... later, possibly in a different program ...
//     auto loaded = load_features("sol_mfcc.feat");
//
// File layout:
//
//     "WFED"                              -- 4-byte magic (weft feature dataset)
//     uint32  version = 1
//     uint32  feature_dim
//     uint32  n_samples
//     uint32  n_classes
//     uint32  type_len; type_len bytes    -- feature type label
//     for each class:                     -- n_classes times
//         uint32 name_len; name_len bytes
//     float[feature_dim * n_samples]      -- column-major, native byte order
//     int32[n_samples]                    -- labels, native byte order
//
// Header integers are little-endian (byte-by-byte) so the metadata can
// be inspected on any host.  The bulk float/int32 payload is in the
// host's native byte order -- if you extract features on one machine
// and load on another with different endianness, the floats will be
// scrambled.  Don't do that.
//
#include "Endian.h"
#include "Matrix.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

struct CachedFeatures {
    Matrix<float>            X;            // feature_dim x n_samples
    std::vector<int>         labels;       // n_samples
    std::vector<std::string> class_names;  // class_names[id] -> name
    std::string              feature_type; // e.g. "mfcc", "logmag"
};

namespace detail {

inline void write_string(std::ofstream& out, const std::string& s) {
    write_le_u32(out, static_cast<std::uint32_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

inline std::string read_string(std::ifstream& in) {
    std::uint32_t len = read_le_u32(in);
    std::string s(len, '\0');
    in.read(s.data(), static_cast<std::streamsize>(len));
    if (!in) throw std::runtime_error("FeatureCache: short read in string");
    return s;
}

constexpr std::uint32_t VERSION = 1;

} // namespace detail

inline void save_features(const std::string& path, const CachedFeatures& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("FeatureCache: cannot open for write: " + path);

    if (data.X.cols() != data.labels.size())
        throw std::runtime_error("FeatureCache: matrix columns != labels");

    out.write("WFED", 4);                                  // magic
    detail::write_le_u32(out, detail::VERSION);
    detail::write_le_u32(out, static_cast<std::uint32_t>(data.X.rows()));
    detail::write_le_u32(out, static_cast<std::uint32_t>(data.X.cols()));
    detail::write_le_u32(out, static_cast<std::uint32_t>(data.class_names.size()));
    detail::write_string(out, data.feature_type);
    for (const auto& name : data.class_names)
        detail::write_string(out, name);

    // Feature matrix: write column-major (matches our examples-as-columns
    // convention) one column at a time, in a small buffer per column.
    const std::size_t feat_dim = data.X.rows();
    const std::size_t n        = data.X.cols();
    std::vector<float> col_buf(feat_dim);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < feat_dim; ++i)
            col_buf[i] = data.X(i, j);
        out.write(reinterpret_cast<const char*>(col_buf.data()),
                  static_cast<std::streamsize>(feat_dim * sizeof(float)));
    }

    // Labels.
    std::vector<std::int32_t> label_buf(data.labels.size());
    for (std::size_t i = 0; i < data.labels.size(); ++i)
        label_buf[i] = static_cast<std::int32_t>(data.labels[i]);
    out.write(reinterpret_cast<const char*>(label_buf.data()),
              static_cast<std::streamsize>(label_buf.size() * sizeof(std::int32_t)));

    if (!out) throw std::runtime_error("FeatureCache: write failed: " + path);
}

inline CachedFeatures load_features(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("FeatureCache: cannot open: " + path);

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "WFED")
        throw std::runtime_error("FeatureCache: bad magic in: " + path);

    std::uint32_t version = detail::read_le_u32(in);
    if (version != detail::VERSION)
        throw std::runtime_error("FeatureCache: unsupported version " +
                                 std::to_string(version) + " in " + path);

    const std::uint32_t feat_dim  = detail::read_le_u32(in);
    const std::uint32_t n_samples = detail::read_le_u32(in);
    const std::uint32_t n_classes = detail::read_le_u32(in);

    CachedFeatures out;
    out.feature_type = detail::read_string(in);
    out.class_names.resize(n_classes);
    for (std::uint32_t i = 0; i < n_classes; ++i)
        out.class_names[i] = detail::read_string(in);

    // Read the feature matrix one column at a time into a buffer, then
    // copy into the Matrix (which has only operator() access).
    out.X = Matrix<float>(feat_dim, n_samples);
    std::vector<float> col_buf(feat_dim);
    for (std::uint32_t j = 0; j < n_samples; ++j) {
        in.read(reinterpret_cast<char*>(col_buf.data()),
                static_cast<std::streamsize>(feat_dim * sizeof(float)));
        if (!in) throw std::runtime_error("FeatureCache: short read in matrix");
        for (std::uint32_t i = 0; i < feat_dim; ++i)
            out.X(i, j) = col_buf[i];
    }

    std::vector<std::int32_t> label_buf(n_samples);
    in.read(reinterpret_cast<char*>(label_buf.data()),
            static_cast<std::streamsize>(label_buf.size() * sizeof(std::int32_t)));
    if (!in) throw std::runtime_error("FeatureCache: short read in labels");
    out.labels.assign(label_buf.begin(), label_buf.end());

    return out;
}

} // namespace weft
