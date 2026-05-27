#pragma once
//
// MNIST.h  --  load MNIST IDX-format files into weft::Matrix and label
//              vectors.
//
// IDX format reminder:
//
//   images:    [0x00000803] [N] [H] [W] [N*H*W bytes of pixels]
//   labels:    [0x00000801] [N] [N bytes of labels]
//
// All 32-bit integers are big-endian; pixels and labels are unsigned bytes.
//
// load_images returns a Matrix<T> of shape (H*W, N) -- one column per
// image, pixels normalised to [0, 1].  This matches our examples-as-columns
// convention so the output is fed straight into a Dense layer.
//
// load_labels returns a std::vector<int> of class indices 0..9.  Pass it
// to one_hot<T>(labels, 10) to get the (10, N) target matrix.
//
//   auto X      = weft::mnist::load_images<float>("train-images-idx3-ubyte");
//   auto labels = weft::mnist::load_labels("train-labels-idx1-ubyte");
//   auto Y      = weft::one_hot<float>(labels, 10);
//
#include "Matrix.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {
namespace mnist {

namespace detail {

// Read 4 big-endian bytes as a uint32_t, regardless of host endianness.
inline std::uint32_t read_be_uint32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("MNIST: unexpected end of file");
    return (std::uint32_t(b[0]) << 24) |
           (std::uint32_t(b[1]) << 16) |
           (std::uint32_t(b[2]) <<  8) |
            std::uint32_t(b[3]);
}

} // namespace detail

template <typename T = float>
Matrix<T> load_images(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("MNIST: cannot open " + path);

    const std::uint32_t magic = detail::read_be_uint32(in);
    if (magic != 0x00000803u)
        throw std::runtime_error("MNIST: not an IDX image file: " + path);

    const std::uint32_t N = detail::read_be_uint32(in);
    const std::uint32_t H = detail::read_be_uint32(in);
    const std::uint32_t W = detail::read_be_uint32(in);
    const std::size_t   F = static_cast<std::size_t>(H) * W;

    Matrix<T> X(F, N);
    std::vector<unsigned char> buf(F);

    for (std::size_t j = 0; j < N; ++j) {
        in.read(reinterpret_cast<char*>(buf.data()), F);
        if (!in) throw std::runtime_error("MNIST: short read on " + path);
        for (std::size_t i = 0; i < F; ++i)
            X(i, j) = static_cast<T>(buf[i]) / T(255);
    }
    return X;
}

inline std::vector<int> load_labels(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("MNIST: cannot open " + path);

    const std::uint32_t magic = detail::read_be_uint32(in);
    if (magic != 0x00000801u)
        throw std::runtime_error("MNIST: not an IDX label file: " + path);

    const std::uint32_t N = detail::read_be_uint32(in);

    std::vector<unsigned char> buf(N);
    in.read(reinterpret_cast<char*>(buf.data()), N);
    if (!in) throw std::runtime_error("MNIST: short read on " + path);

    std::vector<int> labels(N);
    for (std::uint32_t i = 0; i < N; ++i)
        labels[i] = static_cast<int>(buf[i]);
    return labels;
}

} // namespace mnist
} // namespace weft
