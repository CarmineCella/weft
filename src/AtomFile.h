#pragma once
//
// AtomFile.h  --  an atom list on disk.
//
// A matching-pursuit decomposition (see MatchingPursuit.h) is a list of
// atoms plus the three things needed to interpret it: the sample rate, the
// length of the sound, and the dictionary's window lengths.  This is that
// record, with functions to write and read it.
//
// File layout (integers and floats in the host's byte order, like the
// other scratch formats in the library):
//
//     "WATM"
//     u32 sample_rate, u32 n_samples
//     u32 n_window_lengths, then that many u32
//     u32 n_atoms, then per atom:  u32 scale, i32 start, u32 bin,
//                                  f32 amp, f32 phase
//
#include "MatchingPursuit.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

struct AtomFile {
    std::uint32_t            sample_rate = 44100;
    std::uint32_t            n_samples   = 0;
    std::vector<std::size_t> window_lengths;
    std::vector<Atom>        atoms;
};

inline void save_atoms(const std::string& path, const AtomFile& f) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("AtomFile: cannot write " + path);
    auto u32 = [&](std::uint32_t v) { o.write(reinterpret_cast<const char*>(&v), 4); };
    o.write("WATM", 4);
    u32(f.sample_rate);
    u32(f.n_samples);
    u32(static_cast<std::uint32_t>(f.window_lengths.size()));
    for (std::size_t N : f.window_lengths) u32(static_cast<std::uint32_t>(N));
    u32(static_cast<std::uint32_t>(f.atoms.size()));
    for (const Atom& a : f.atoms) {
        u32(a.scale);
        o.write(reinterpret_cast<const char*>(&a.start), 4);
        u32(a.bin);
        o.write(reinterpret_cast<const char*>(&a.amp), 4);
        o.write(reinterpret_cast<const char*>(&a.phase), 4);
    }
    if (!o) throw std::runtime_error("AtomFile: write failed: " + path);
}

inline AtomFile load_atoms(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("AtomFile: cannot open " + path);
    auto u32 = [&]() { std::uint32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); return v; };
    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "WATM")
        throw std::runtime_error("AtomFile: not an atom file: " + path);
    AtomFile f;
    f.sample_rate = u32();
    f.n_samples   = u32();
    const std::uint32_t ns = u32();
    for (std::uint32_t i = 0; i < ns; ++i) f.window_lengths.push_back(u32());
    const std::uint32_t na = u32();
    f.atoms.resize(na);
    for (Atom& a : f.atoms) {
        a.scale = u32();
        in.read(reinterpret_cast<char*>(&a.start), 4);
        a.bin = u32();
        in.read(reinterpret_cast<char*>(&a.amp), 4);
        in.read(reinterpret_cast<char*>(&a.phase), 4);
    }
    if (!in) throw std::runtime_error("AtomFile: short read in " + path);
    return f;
}

} // namespace weft
