#pragma once
//
// CIFAR10.h  --  loader for the CIFAR-10 binary dataset.
//
// The binary version of CIFAR-10 stores each image as a 3073-byte
// record:
//
//   byte 0:        label (0-9)
//   bytes 1-1024:  R plane (1024 = 32 rows of 32 R values, row-major)
//   bytes 1025-2048: G plane
//   bytes 2049-3072: B plane
//
// Each .bin file holds 10,000 such records.  The official tarball gives
// you five training batches and one test batch.
//
// The byte order "all R then all G then all B, each row-major" is
// EXACTLY our NCHW layout for one image: channel slowest, height next,
// width fastest.  So once the 1-byte label has been read, the next
// 3072 bytes can be copied straight into the flat data buffer of a
// Tensor4D<T>(N, 3, 32, 32) at the correct image offset, with no
// per-channel rearrangement needed.  The only transformation is to
// scale from uint8 [0, 255] to T [0, 1].
//
// Usage:
//     auto train = load_cifar10<float>(cifar10_train_paths("data/cifar-10-batches-bin"));
//     auto test  = load_cifar10<float>(cifar10_test_paths ("data/cifar-10-batches-bin"));
//     // train.images is Tensor4D<float>(50000, 3, 32, 32)
//     // train.labels is std::vector<int> of length 50000
//

#include "Tensor4D.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

template <typename T = float>
struct CIFAR10Data {
    Tensor4D<T>       images;     // (N, 3, 32, 32), normalised to [0, 1]
    std::vector<int>  labels;     // N labels, each 0..9
};

template <typename T = float>
inline CIFAR10Data<T> load_cifar10(const std::vector<std::string>& paths) {
    constexpr std::size_t IMG_BYTES = 3u * 32u * 32u;   // 3072
    constexpr std::size_t REC_BYTES = 1u + IMG_BYTES;    // 3073

    // ---- first pass: tally how many records each file contains ----
    std::size_t total = 0;
    std::vector<std::size_t> counts;
    counts.reserve(paths.size());
    for (const auto& path : paths) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("CIFAR10: cannot open " + path);
        const std::streamsize size = f.tellg();
        if (size < 0)
            throw std::runtime_error("CIFAR10: stat failed for " + path);
        const std::size_t bytes = static_cast<std::size_t>(size);
        if (bytes == 0 || bytes % REC_BYTES != 0)
            throw std::runtime_error("CIFAR10: " + path +
                " is not a multiple of 3073 bytes (looks malformed)");
        counts.push_back(bytes / REC_BYTES);
        total += counts.back();
    }

    CIFAR10Data<T> out;
    out.images = Tensor4D<T>(total, 3, 32, 32);
    out.labels.assign(total, 0);

    // ---- second pass: actually read the records ----
    T* img_data = out.images.data();
    std::size_t n = 0;
    std::vector<std::uint8_t> buf(IMG_BYTES);

    for (std::size_t fi = 0; fi < paths.size(); ++fi) {
        std::ifstream f(paths[fi], std::ios::binary);
        if (!f) throw std::runtime_error("CIFAR10: cannot open " + paths[fi]);

        for (std::size_t i = 0; i < counts[fi]; ++i) {
            std::uint8_t label_byte;
            f.read(reinterpret_cast<char*>(&label_byte), 1);
            if (!f) throw std::runtime_error("CIFAR10: unexpected EOF in " + paths[fi]);
            out.labels[n] = static_cast<int>(label_byte);

            f.read(reinterpret_cast<char*>(buf.data()),
                   static_cast<std::streamsize>(IMG_BYTES));
            if (!f) throw std::runtime_error("CIFAR10: unexpected EOF in " + paths[fi]);

            // Direct copy: CIFAR's byte order == our NCHW storage.
            T* dest = img_data + n * IMG_BYTES;
            for (std::size_t k = 0; k < IMG_BYTES; ++k)
                dest[k] = static_cast<T>(buf[k]) / T(255);
            ++n;
        }
    }
    return out;
}

// Convenience: the standard file lists.  Pass these the path to the
// cifar-10-batches-bin/ directory produced by download_cifar10.sh.
inline std::vector<std::string> cifar10_train_paths(const std::string& dir) {
    return {
        dir + "/data_batch_1.bin",
        dir + "/data_batch_2.bin",
        dir + "/data_batch_3.bin",
        dir + "/data_batch_4.bin",
        dir + "/data_batch_5.bin",
    };
}

inline std::vector<std::string> cifar10_test_paths(const std::string& dir) {
    return { dir + "/test_batch.bin" };
}

// Canonical class names, in label-index order.
inline const char* cifar10_class_name(int label) {
    static const char* names[10] = {
        "airplane", "automobile", "bird",   "cat",  "deer",
        "dog",      "frog",       "horse",  "ship", "truck"
    };
    return (label >= 0 && label < 10) ? names[label] : "?";
}

} // namespace weft
