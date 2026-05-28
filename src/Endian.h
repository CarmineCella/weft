#pragma once
//
// Endian.h -- small helpers for reading and writing fixed-width integers
//             in a specified byte order, regardless of host endianness.
//
// All assemble or disassemble bytes by hand rather than reinterpret-casting
// a raw int, so they're correct on big- or little-endian hosts.  The
// helpers are inline and stateless; sharing them across modules avoids the
// "every parser has its own copy of read_le_u32" duplication.
//
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace weft {
namespace detail {

// ---- Little-endian reads ----

inline std::uint16_t read_le_u16(std::ifstream& in) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    if (!in) throw std::runtime_error("Endian: unexpected end of file");
    return std::uint16_t(b[0]) | (std::uint16_t(b[1]) << 8);
}

inline std::uint32_t read_le_u32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Endian: unexpected end of file");
    return  std::uint32_t(b[0])
         | (std::uint32_t(b[1]) <<  8)
         | (std::uint32_t(b[2]) << 16)
         | (std::uint32_t(b[3]) << 24);
}

// ---- Little-endian writes ----

inline void write_le_u16(std::ofstream& out, std::uint16_t x) {
    unsigned char b[2] = {
        static_cast<unsigned char>( x       & 0xff),
        static_cast<unsigned char>((x >> 8) & 0xff)
    };
    out.write(reinterpret_cast<char*>(b), 2);
}

inline void write_le_u32(std::ofstream& out, std::uint32_t x) {
    unsigned char b[4] = {
        static_cast<unsigned char>( x        & 0xff),
        static_cast<unsigned char>((x >>  8) & 0xff),
        static_cast<unsigned char>((x >> 16) & 0xff),
        static_cast<unsigned char>((x >> 24) & 0xff)
    };
    out.write(reinterpret_cast<char*>(b), 4);
}

} // namespace detail
} // namespace weft
