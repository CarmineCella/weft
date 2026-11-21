#pragma once
//
// Shared checkpoint header. Version 2 records scalar width and exact
// configuration; version 1 remains readable for existing Network/ConvNetwork
// weight files, whose architecture must still be supplied by the caller.
//
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace weft {
namespace detail {

template <typename T>
void write_checkpoint_header(std::ostream& out, const char* magic,
                             std::uint32_t count, const std::string& config) {
    const std::uint32_t version = 2, scalar_bytes = sizeof(T);
    const std::uint32_t length = static_cast<std::uint32_t>(config.size());
    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&scalar_bytes), sizeof(scalar_bytes));
    out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    out.write(config.data(), static_cast<std::streamsize>(config.size()));
    if (!out) throw std::runtime_error("Checkpoint: header write failed");
}

template <typename T>
void read_checkpoint_header(std::istream& in, const char* expected_magic,
                            std::uint32_t expected_count,
                            const std::string& expected_config,
                            bool allow_version_one = true) {
    char magic[4] = {};
    std::uint32_t version = 0, count = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in) throw std::runtime_error("Checkpoint: truncated header");
    if (std::string(magic, 4) != std::string(expected_magic, 4))
        throw std::runtime_error("Checkpoint: bad magic");
    if (version != 2 && !(allow_version_one && version == 1))
        throw std::runtime_error("Checkpoint: unsupported version " + std::to_string(version));
    if (count != expected_count)
        throw std::runtime_error("Checkpoint: layer count mismatch");
    if (version == 1) return;

    std::uint32_t scalar_bytes = 0, length = 0;
    in.read(reinterpret_cast<char*>(&scalar_bytes), sizeof(scalar_bytes));
    in.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (!in) throw std::runtime_error("Checkpoint: truncated configuration header");
    if (scalar_bytes != sizeof(T))
        throw std::runtime_error("Checkpoint: scalar type size mismatch");
    if (length != expected_config.size())
        throw std::runtime_error("Checkpoint: configuration mismatch");
    std::string config(length, '\0');
    in.read(config.data(), static_cast<std::streamsize>(length));
    if (!in) throw std::runtime_error("Checkpoint: truncated configuration");
    if (config != expected_config)
        throw std::runtime_error("Checkpoint: configuration mismatch");
}

} // namespace detail
} // namespace weft
