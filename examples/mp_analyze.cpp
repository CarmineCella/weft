// mp_analyze / mp_analyze.cpp
//
// Decompose a sound into time-frequency atoms with matching pursuit, and
// listen to what the atoms keep and what they leave out.
//
// There is no neural network here.  This program is the "codec" half of
// the final system: it turns a waveform into a list of atoms and a list
// of atoms back into a waveform.  The transformer of the next chapter
// works on such lists.
//
// Usage:
//   mp_analyze decompose <in.wav> <prefix> [n_atoms]
//       writes  <prefix>.atoms              the atom list
//               <prefix>_<N>atoms.wav       resynthesis from the first N
//                                           atoms, for several N
//               <prefix>_residual.wav       what all the atoms left out
//       and prints how the signal-to-residual ratio grows with N
//   mp_analyze render <in.atoms> <out.wav> [mode] [value]
//       rebuild from an atom list.  mode is one of:
//         all                   every atom (default)
//         first <N>             only the first N atoms (the strongest)
//         long                  only the two longest window lengths:
//                               the steady, tonal part
//         short                 only the two shortest: transients and noise
//         stretch <factor>      move the atoms apart in time by <factor>
//                               without changing them (time stretching)
//         transpose <semitones> move every atom's frequency
//   mp_analyze info <in.atoms>
//       statistics of an atom list

#include "MatchingPursuit.h"
#include "Wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

namespace {

// An atom list with what is needed to interpret it.
struct AtomFile {
    std::uint32_t            sample_rate = 44100;
    std::uint32_t            n_samples   = 0;
    std::vector<std::size_t> window_lengths;
    std::vector<Atom>        atoms;
};

void save_atoms(const std::string& path, const AtomFile& f) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + path);
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
}

AtomFile load_atoms(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    auto u32 = [&]() { std::uint32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); return v; };
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "WATM") throw std::runtime_error("not an atom file: " + path);
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
    if (!in) throw std::runtime_error("short read in " + path);
    return f;
}

int cmd_decompose(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: mp_analyze decompose <in.wav> <prefix> [n_atoms]\n";
        return 1;
    }
    using clock = std::chrono::steady_clock;
    const std::string in_path = argv[2], prefix = argv[3];
    const std::size_t n_atoms = (argc > 4) ? std::size_t(std::atoi(argv[4])) : 8000;

    const WavData wav = load_wav(in_path);
    const std::vector<float>& x = wav.samples;
    MatchingPursuit<float> mp;

    std::cout << "weft :: mp_analyze decompose\n  input: " << in_path << "  ("
              << x.size() << " samples @ " << wav.sample_rate << " Hz, "
              << std::fixed << std::setprecision(2) << x.size() / double(wav.sample_rate)
              << " s)\n  window lengths:";
    for (std::size_t N : mp.window_lengths())
        std::cout << " " << N << " (" << std::setprecision(1)
                  << 1000.0 * N / wav.sample_rate << " ms)";
    std::cout << "\n\n";

    const auto t0 = clock::now();
    std::vector<double> trace;
    std::vector<float>  residual;
    AtomFile f;
    f.sample_rate    = wav.sample_rate;
    f.n_samples      = static_cast<std::uint32_t>(x.size());
    f.window_lengths = mp.window_lengths();
    f.atoms          = mp.decompose(x, n_atoms, 0.0f, &trace, &residual);
    const double secs = std::chrono::duration<double>(clock::now() - t0).count();

    // How much each prefix of the list explains.  The ratio is the energy
    // of the signal over the energy of what is left, in dB.
    std::cout << " atoms   atoms/s   signal-to-residual   energy explained\n"
              << " -----   -------   ------------------   ----------------\n";
    const double dur = x.size() / double(wav.sample_rate);
    std::vector<std::size_t> marks;
    for (std::size_t n : {250u, 500u, 1000u, 2000u, 4000u, 8000u, 16000u, 32000u})
        if (n < f.atoms.size()) marks.push_back(n);
    marks.push_back(f.atoms.size());
    for (std::size_t n : marks) {
        const double ratio = trace[n] / trace[0];
        std::cout << std::setw(6) << n << "   " << std::setw(7) << std::setprecision(0)
                  << n / dur << "   " << std::setw(15) << std::setprecision(1)
                  << -10.0 * std::log10(std::max(ratio, 1e-12)) << " dB   "
                  << std::setw(14) << std::setprecision(1) << 100.0 * (1.0 - ratio) << "%\n";
    }
    std::cout << "\n  " << f.atoms.size() << " atoms in " << std::setprecision(1) << secs
              << " s;  the list holds " << std::setprecision(1)
              << 100.0 * 5.0 * f.atoms.size() / double(x.size())
              << "% as many numbers as the waveform\n";

    save_atoms(prefix + ".atoms", f);
    for (std::size_t n : marks) {
        if (n != f.atoms.size() && n != 500 && n != 2000) continue;
        std::vector<Atom> part(f.atoms.begin(), f.atoms.begin() + long(n));
        save_wav(prefix + "_" + std::to_string(n) + "atoms.wav",
                 mp.synthesize(part, x.size()), wav.sample_rate);
    }
    save_wav(prefix + "_residual.wav", residual, wav.sample_rate);
    std::cout << "  wrote " << prefix << ".atoms, resyntheses and residual\n";
    return 0;
}

int cmd_render(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: mp_analyze render <in.atoms> <out.wav> [mode] [value]\n";
        return 1;
    }
    AtomFile f = load_atoms(argv[2]);
    const std::string mode = (argc > 4) ? argv[4] : "all";
    const double value = (argc > 5) ? std::atof(argv[5]) : 0.0;
    MatchingPursuit<float> mp(f.window_lengths);
    std::size_t n_samples = f.n_samples;
    std::vector<Atom> atoms;

    const std::size_t n_scales = f.window_lengths.size();
    if (mode == "all") {
        atoms = f.atoms;
    } else if (mode == "first") {
        const std::size_t n = std::min(f.atoms.size(), std::size_t(value));
        atoms.assign(f.atoms.begin(), f.atoms.begin() + long(n));
    } else if (mode == "long" || mode == "short") {
        // "long" keeps the upper half of the window lengths, "short" the lower.
        for (const Atom& a : f.atoms) {
            const bool is_long = a.scale >= n_scales / 2;
            if (is_long == (mode == "long")) atoms.push_back(a);
        }
    } else if (mode == "stretch") {
        // Move each atom's centre; the atom itself (length, frequency,
        // amplitude) is unchanged, so pitch and timbre are preserved.  The
        // phase is advanced so that a partial which continued from one atom
        // into the next still lines up at the new distance.
        const double factor = (value > 0.0) ? value : 2.0;
        n_samples = std::size_t(double(n_samples) * factor);
        for (Atom a : f.atoms) {
            const double N = double(f.window_lengths[a.scale]);
            const double centre = double(a.start) + 0.5 * N;
            const double moved  = centre * factor;
            const long new_start = std::lround(moved - 0.5 * N);
            const double shift = double(new_start - a.start);
            a.phase = static_cast<float>(std::fmod(
                double(a.phase) + 2.0 * 3.14159265358979 * double(a.bin) / N *
                (shift - shift / factor), 2.0 * 3.14159265358979));
            a.start = static_cast<std::int32_t>(new_start);
            atoms.push_back(a);
        }
    } else if (mode == "transpose") {
        const double ratio = std::pow(2.0, value / 12.0);
        for (Atom a : f.atoms) {
            const double N = double(f.window_lengths[a.scale]);
            const long bin = std::lround(double(a.bin) * ratio);
            if (bin < 1 || bin >= long(N / 2)) continue;       // out of range
            a.bin = static_cast<std::uint32_t>(bin);
            atoms.push_back(a);
        }
    } else {
        std::cerr << "unknown mode '" << mode << "'\n";
        return 1;
    }

    save_wav(argv[3], mp.synthesize(atoms, n_samples), f.sample_rate);
    std::cout << "wrote " << argv[3] << "  (" << atoms.size() << " of " << f.atoms.size()
              << " atoms, mode " << mode << ")\n";
    return 0;
}

int cmd_info(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: mp_analyze info <in.atoms>\n"; return 1; }
    const AtomFile f = load_atoms(argv[2]);
    std::cout << f.atoms.size() << " atoms, " << std::fixed << std::setprecision(2)
              << f.n_samples / double(f.sample_rate) << " s @ " << f.sample_rate << " Hz\n"
              << "window    atoms    share of atom energy\n";
    std::vector<std::size_t> count(f.window_lengths.size(), 0);
    std::vector<double> en(f.window_lengths.size(), 0.0);
    double total = 0.0;
    for (const Atom& a : f.atoms) {
        ++count[a.scale];
        en[a.scale] += double(a.amp) * a.amp;
        total += double(a.amp) * a.amp;
    }
    for (std::size_t s = 0; s < f.window_lengths.size(); ++s)
        std::cout << std::setw(6) << f.window_lengths[s] << "   " << std::setw(6) << count[s]
                  << "    " << std::setprecision(1) << 100.0 * en[s] / std::max(total, 1e-30)
                  << "%\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = (argc > 1) ? argv[1] : "";
    try {
        if (cmd == "decompose") return cmd_decompose(argc, argv);
        if (cmd == "render")    return cmd_render(argc, argv);
        if (cmd == "info")      return cmd_info(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage:\n"
                 "  mp_analyze decompose <in.wav> <prefix> [n_atoms]\n"
                 "  mp_analyze render    <in.atoms> <out.wav> [all|first N|long|short|"
                 "stretch F|transpose S]\n"
                 "  mp_analyze info      <in.atoms>\n";
    return 1;
}
