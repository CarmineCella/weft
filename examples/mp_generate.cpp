// mp_generate / mp_generate.cpp
//
// A generative model of sound built from the two halves of this book:
// matching pursuit turns recordings into lists of atoms (mp_analyze), and
// a transformer learns to continue such lists.  Sampling from it writes a
// new list of atoms, and the atoms are rendered back into audio.
//
// The network never produces a waveform or a spectrum.  It decides WHICH
// atom comes next -- when, how long, at what frequency, how loud -- and
// the sound is assembled from the same windowed sinusoids matching
// pursuit found in the recordings.
//
// TOKENS.  The atoms of a sound are sorted by the time of their centre
// and each becomes five tokens:
//
//     DT     time since the previous atom's centre: 96 values in exact
//            steps of 32 samples (0.7 ms, up to 70 ms), then 32 values
//            growing geometrically up to 3 s, for the silences
//     SCALE  which of the four window lengths
//     FREQ   frequency on a logarithmic grid, 60 steps per octave
//     AMP    level in steps of 1.5 dB below the loudest atom of the sound
//     PHASE  phase of the cosine at the centre of the atom, 16 values
//
// Phase was not a token at first; it was restored at synthesis by a rule.
// Measuring the round trip atoms -> tokens -> atoms showed that this was
// the wrong economy: the token grid for time, frequency and level changes
// the spectrogram by 6-15%, while replacing the phases, by any rule we
// tried, changes it by 40-47%.  The roundtrip command still offers the
// rules ("locked", "random") so that the difference can be heard.
//
// Usage:
//   mp_generate roundtrip <in.atoms> <out.wav> [n_atoms] [phase]
//       atoms -> tokens -> atoms -> audio, with no model: what the token
//       grid alone costs.  phase: token (default) | original | locked | random
//   mp_generate train <prefix> <epochs> <n_atoms> <a.atoms> [b.atoms ...]
//       n_atoms = how many atoms (the strongest) to keep from each file
//   mp_generate generate <prefix> <out.wav> [seconds] [temperature] [seed]
//                        [prime.atoms] [prime_seconds]
//       sample a new atom list and render it; optionally start from the
//       opening of a real sound and let the model continue it
//   mp_generate markov <prefix> <out.wav> [seconds] [seed]
//       the baseline: a first-order Markov chain over atoms
//   mp_generate stats <n_atoms> <a.atoms> [b.atoms ...]
//       how the atoms of each sound are organised in time

#include "Adam.h"
#include "AtomFile.h"
#include "CrossEntropy.h"
#include "MatchingPursuit.h"
#include "Matrix.h"
#include "TokenTransformer.h"
#include "Wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace weft;

namespace {

// ---- vocabulary ----------------------------------------------------------
constexpr int DT_UNIT   = 32;      // samples per time step
constexpr int DT_N      = 128;
constexpr int SCALE_N   = 4;
constexpr int FREQ_N    = 608;     // 60 steps per octave above 20 Hz
constexpr int AMP_N     = 48;      // 1.5 dB steps
constexpr int PHASE_N   = 16;
constexpr int DT_BASE    = 0;
constexpr int SCALE_BASE = DT_BASE + DT_N;
constexpr int FREQ_BASE  = SCALE_BASE + SCALE_N;
constexpr int AMP_BASE   = FREQ_BASE + FREQ_N;
constexpr int PHASE_BASE = AMP_BASE + AMP_N;
constexpr int TOK_BOS    = PHASE_BASE + PHASE_N;
constexpr int TOK_EOS    = TOK_BOS + 1;
constexpr int VOCAB      = TOK_EOS + 1;
constexpr int PER_ATOM   = 5;

const double FREQ_MIN = 20.0, STEPS_PER_OCTAVE = 60.0, AMP_STEP_DB = 1.5;

// The time gaps a DT token can express, in units of DT_UNIT samples.
// Most atoms follow the previous one within a few milliseconds, and there
// the grid is exact.  But a sound also has silences, and a linear grid
// that reached 3 s would need 4000 tokens.  An earlier version simply
// capped the gap at 127 steps (92 ms): every pause was shortened to that,
// and everything after a pause arrived too early.
constexpr int DT_LINEAR = 96;
inline int dt_value(int token) {
    if (token < DT_LINEAR) return token;
    return int(std::lround(double(DT_LINEAR) *
                           std::pow(4096.0 / DT_LINEAR, double(token - DT_LINEAR + 1) /
                                                        double(DT_N - DT_LINEAR))));
}
inline int dt_token(long gap) {
    if (gap < DT_LINEAR) return int(std::max<long>(0, gap));
    int best = DT_LINEAR - 1; long best_err = gap - (DT_LINEAR - 1);
    for (int t = DT_LINEAR; t < DT_N; ++t) {
        const long err = std::labs(long(dt_value(t)) - gap);
        if (err < best_err) { best_err = err; best = t; }
    }
    return best;
}

// ---- model size ----------------------------------------------------------
constexpr std::size_t D_MODEL  = 128;
constexpr std::size_t N_HEADS  = 4;
constexpr std::size_t N_LAYERS = 3;
constexpr std::size_t D_FF     = 4 * D_MODEL;
constexpr std::size_t BLOCK    = 320;     // tokens per training window = 64 atoms

// ---- atoms <-> tokens ----------------------------------------------------

AtomFile load_checked(const std::string& path) {
    AtomFile f = load_atoms(path);
    if (f.window_lengths.size() != SCALE_N)
        throw std::runtime_error("atom file does not use 4 window lengths: " + path);
    return f;
}

long centre_of(const Atom& a, const std::vector<std::size_t>& sizes) {
    return long(a.start) + long(sizes[a.scale] / 2);
}

// The strongest n atoms of a file, sorted by centre time (ties by frequency).
std::vector<Atom> select_atoms(const AtomFile& f, std::size_t n) {
    std::vector<Atom> atoms(f.atoms.begin(),
                            f.atoms.begin() + long(std::min(n, f.atoms.size())));
    std::stable_sort(atoms.begin(), atoms.end(), [&](const Atom& a, const Atom& b) {
        const long ca = centre_of(a, f.window_lengths), cb = centre_of(b, f.window_lengths);
        if (ca != cb) return ca < cb;
        return double(a.bin) / double(f.window_lengths[a.scale]) <
               double(b.bin) / double(f.window_lengths[b.scale]);
    });
    return atoms;
}

// An atom stores its phase at its first sample.  The token stores it at
// the CENTRE of the atom instead: phase_centre = phase_start + pi * bin.
// The frequency grid can move an atom to a neighbouring bin, and the
// centre is where the window is largest, so that is where the cosine has
// to stay put.
inline double two_pi() { return 6.283185307179586; }
inline double wrap(double x) { x = std::fmod(x, two_pi()); return x < 0 ? x + two_pi() : x; }

// BOS, then five tokens per atom, then EOS.
std::vector<int> atoms_to_tokens(const std::vector<Atom>& atoms, const AtomFile& f) {
    float max_amp = 0.0f;
    for (const Atom& a : atoms) max_amp = std::max(max_amp, std::fabs(a.amp));
    std::vector<int> toks = {TOK_BOS};
    long prev = atoms.empty() ? 0 : centre_of(atoms[0], f.window_lengths);
    for (const Atom& a : atoms) {
        const long c = centre_of(a, f.window_lengths);
        // `prev` advances by what the token can express, not by the true
        // gap, so that rounding errors in long gaps do not accumulate.
        const int dt = dt_token((c - prev) / DT_UNIT);
        prev += long(dt_value(dt)) * DT_UNIT;
        const double hz = double(a.bin) * f.sample_rate / double(f.window_lengths[a.scale]);
        const int fi = int(std::lround(STEPS_PER_OCTAVE * std::log2(std::max(hz, FREQ_MIN) / FREQ_MIN)));
        const double db = 20.0 * std::log10(std::max(std::fabs(a.amp), 1e-9f) / max_amp);
        const int ai = int(std::lround(-db / AMP_STEP_DB));
        toks.push_back(DT_BASE + dt);
        toks.push_back(SCALE_BASE + int(a.scale));
        toks.push_back(FREQ_BASE + std::min(FREQ_N - 1, std::max(0, fi)));
        toks.push_back(AMP_BASE + std::min(AMP_N - 1, std::max(0, ai)));
        // A negative coefficient is the same atom half a cycle later.
        const double pc = wrap(double(a.phase) + 3.14159265358979 * double(a.bin)
                               + (a.amp < 0 ? 3.14159265358979 : 0.0));
        toks.push_back(PHASE_BASE + int(std::lround(pc / two_pi() * PHASE_N)) % PHASE_N);
    }
    toks.push_back(TOK_EOS);
    return toks;
}

// How a rebuilt atom gets its phase.
//   locked : every atom is a slice of one endless cosine at its frequency,
//            cos(2 pi f t) with t counted from the start of the sound.  Two
//            atoms at the same frequency that overlap in time then add up
//            instead of cancelling, which is what a sustained partial
//            needs.
//   random : uniformly random.
float phase_for(const Atom& a, const std::vector<std::size_t>& sizes,
                const std::string& mode, std::mt19937& rng) {
    const double two_pi = 6.283185307179586;
    if (mode == "random")
        return float(std::uniform_real_distribution<double>(0.0, two_pi)(rng));
    const double N = double(sizes[a.scale]);
    return float(std::fmod(two_pi * double(a.bin) / N * double(a.start), two_pi));
}

// Tokens -> atoms.  Tokens outside their field's range end the list.
std::vector<Atom> tokens_to_atoms(const std::vector<int>& toks,
                                  const std::vector<std::size_t>& sizes,
                                  double sample_rate, const std::string& phase_mode,
                                  unsigned seed, long first_centre = 4096)
{
    std::mt19937 rng(seed);
    std::vector<Atom> atoms;
    long centre = first_centre;
    std::size_t i = (!toks.empty() && toks[0] == TOK_BOS) ? 1 : 0;
    for (; i + PER_ATOM <= toks.size(); i += PER_ATOM) {
        const int dt = toks[i] - DT_BASE, sc = toks[i + 1] - SCALE_BASE,
                  fi = toks[i + 2] - FREQ_BASE, ai = toks[i + 3] - AMP_BASE,
                  pi = toks[i + 4] - PHASE_BASE;
        if (dt < 0 || dt >= DT_N || sc < 0 || sc >= SCALE_N ||
            fi < 0 || fi >= FREQ_N || ai < 0 || ai >= AMP_N ||
            pi < 0 || pi >= PHASE_N) break;
        centre += long(dt_value(dt)) * DT_UNIT;
        const double N  = double(sizes[std::size_t(sc)]);
        const double hz = FREQ_MIN * std::pow(2.0, double(fi) / STEPS_PER_OCTAVE);
        long bin = std::lround(hz * N / sample_rate);
        bin = std::min<long>(long(N / 2) - 1, std::max<long>(1, bin));
        Atom a;
        a.scale = std::uint32_t(sc);
        a.start = std::int32_t(centre - long(N / 2));
        a.bin   = std::uint32_t(bin);
        a.amp   = float(std::pow(10.0, -double(ai) * AMP_STEP_DB / 20.0));
        if (phase_mode == "token")
            a.phase = float(wrap(double(pi) * two_pi() / PHASE_N
                                 - 3.14159265358979 * double(bin)));
        else
            a.phase = phase_for(a, sizes, phase_mode, rng);
        atoms.push_back(a);
    }
    return atoms;
}

// Render atoms and scale so that the peak is 0.9.
std::vector<float> render(const std::vector<Atom>& atoms,
                          const std::vector<std::size_t>& sizes, std::size_t n_samples) {
    MatchingPursuit<float> mp(sizes);
    auto y = mp.synthesize(atoms, n_samples);
    float peak = 0.0f;
    for (float v : y) peak = std::max(peak, std::fabs(v));
    if (peak > 0.0f) for (float& v : y) v *= 0.9f / peak;
    return y;
}

// Which field the token at stream index i belongs to (index 0 is BOS).
inline int field_of(std::size_t i) { return int((i + PER_ATOM - 1) % PER_ATOM); }   // 0=DT 1=SCALE 2=FREQ 3=AMP 4=PHASE
inline void field_range(int field, int& lo, int& hi) {
    switch (field) {
        case 0:  lo = DT_BASE;    hi = DT_BASE + DT_N;       break;
        case 1:  lo = SCALE_BASE; hi = SCALE_BASE + SCALE_N; break;
        case 2:  lo = FREQ_BASE;  hi = FREQ_BASE + FREQ_N;   break;
        case 3:  lo = AMP_BASE;   hi = AMP_BASE + AMP_N;     break;
        default: lo = PHASE_BASE; hi = PHASE_BASE + PHASE_N; break;
    }
}

Matrix<float> one_hot_seq(const std::vector<int>& ids) {
    Matrix<float> Y(VOCAB, ids.size(), 0.0f);
    for (std::size_t t = 0; t < ids.size(); ++t) Y(std::size_t(ids[t]), t) = 1.0f;
    return Y;
}

struct Model : TokenTransformer<float> {
    Model() : TokenTransformer<float>(VOCAB, D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK) {}
};

// The atom list of a generated sound, next to its .wav, so that it can be
// inspected with `mp_generate stats` or re-rendered with mp_analyze.
void save_generated(const std::string& wav_path, const std::vector<Atom>& atoms,
                    const std::vector<std::size_t>& sizes, double sample_rate,
                    std::size_t n_samples) {
    AtomFile f;
    f.sample_rate    = std::uint32_t(sample_rate);
    f.n_samples      = std::uint32_t(n_samples);
    f.window_lengths = sizes;
    f.atoms          = atoms;
    std::string path = wav_path;
    const std::size_t dot = path.rfind('.');
    if (dot != std::string::npos) path = path.substr(0, dot);
    save_atoms(path + ".atoms", f);
}

// Read <prefix>.txt: sample rate, window lengths, training token streams.
bool load_meta(const std::string& prefix, double& sample_rate,
               std::vector<std::size_t>& sizes, std::vector<std::vector<int>>& train) {
    std::ifstream meta(prefix + ".txt");
    if (!meta) { std::cerr << "cannot open " << prefix << ".txt\n"; return false; }
    meta >> sample_rate;
    sizes.assign(SCALE_N, 0);
    for (auto& N : sizes) meta >> N;
    std::string line;
    std::getline(meta, line);
    while (std::getline(meta, line)) {
        std::vector<int> s; std::size_t pos = 0;
        while (pos < line.size()) {
            const std::size_t sp = line.find(' ', pos);
            if (sp == std::string::npos) break;
            s.push_back(std::stoi(line.substr(pos, sp - pos)));
            pos = sp + 1;
        }
        if (!s.empty()) train.push_back(std::move(s));
    }
    return true;
}

// ---- commands ------------------------------------------------------------

int cmd_roundtrip(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: mp_generate roundtrip <in.atoms> <out.wav> [n_atoms] [phase]\n";
        return 1;
    }
    const AtomFile f = load_checked(argv[2]);
    const std::size_t n = (argc > 4) ? std::size_t(std::atoi(argv[4])) : 2000;
    const std::string mode = (argc > 5) ? argv[5] : "token";
    const std::vector<Atom> src = select_atoms(f, n);
    const std::vector<int> toks = atoms_to_tokens(src, f);
    std::vector<Atom> out = tokens_to_atoms(toks, f.window_lengths, f.sample_rate, mode, 1,
                                            src.empty() ? 0 : centre_of(src[0], f.window_lengths));
    if (mode == "original")
        for (std::size_t i = 0; i < out.size() && i < src.size(); ++i) {
            // The measured phase, unquantised, kept at the atom's centre.
            const double pc = double(src[i].phase) + 3.14159265358979 * double(src[i].bin)
                              + (src[i].amp < 0 ? 3.14159265358979 : 0.0);
            out[i].phase = float(wrap(pc - 3.14159265358979 * double(out[i].bin)));
        }
    save_wav(argv[3], render(out, f.window_lengths, f.n_samples), f.sample_rate);
    std::cout << "wrote " << argv[3] << "  (" << out.size() << " atoms through "
              << toks.size() << " tokens, phase " << mode << ")\n";
    return 0;
}

int cmd_train(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: mp_generate train <prefix> <epochs> <n_atoms> <a.atoms> [b.atoms ...]\n";
        return 1;
    }
    using clock = std::chrono::steady_clock;
    const std::string prefix = argv[2];
    const int epochs = std::atoi(argv[3]);
    const std::size_t n_atoms = std::size_t(std::atoi(argv[4]));

    // Each file gives one token sequence; its last tenth (in time) is held out.
    std::vector<std::vector<int>> train, held;
    std::vector<std::size_t> sizes;
    std::uint32_t sample_rate = 44100;
    std::cout << "weft :: mp_generate train\n";
    for (int i = 5; i < argc; ++i) {
        const AtomFile f = load_checked(argv[i]);
        sizes = f.window_lengths;
        sample_rate = f.sample_rate;
        const std::vector<int> toks = atoms_to_tokens(select_atoms(f, n_atoms), f);
        const std::size_t n = (toks.size() - 2) / PER_ATOM;         // atoms
        const std::size_t cut = 1 + PER_ATOM * (n - n / 10);        // aligned to an atom
        std::vector<int> a(toks.begin(), toks.begin() + long(cut));
        std::vector<int> b(toks.begin() + long(cut) - 1, toks.end() - 1);
        b[0] = TOK_BOS;                                             // keeps the alignment
        std::cout << "  " << argv[i] << ": " << n << " atoms, " << a.size()
                  << " training tokens, " << b.size() << " held out\n";
        train.push_back(std::move(a));
        held.push_back(std::move(b));
    }

    // Two baselines, fitted on the training tokens and scored on the
    // held-out ones (average cross-entropy, nats per token).  Both know
    // which field each position holds, as the transformer does.
    //   unigram : how often each value of a field occurs
    //   Markov  : each field given the same field of the previous atom --
    //             the model the `markov` command samples from
    {
        std::vector<double> uni(VOCAB, 1.0), big(std::size_t(VOCAB) * VOCAB, 1.0);
        for (const auto& seq : train)
            for (std::size_t i = 1; i + 1 < seq.size(); ++i) {
                uni[std::size_t(seq[i])] += 1.0;
                if (i > PER_ATOM)
                    big[std::size_t(seq[i - PER_ATOM]) * VOCAB + std::size_t(seq[i])] += 1.0;
            }
        double h_uni = 0.0, h_big = 0.0; std::size_t n = 0;
        for (const auto& seq : held)
            for (std::size_t i = 1 + PER_ATOM; i < seq.size(); ++i) {
                int lo, hi;
                field_range(field_of(i), lo, hi);
                double tu = 0.0, tb = 0.0;
                const double* row = &big[std::size_t(seq[i - PER_ATOM]) * VOCAB];
                for (int v = lo; v < hi; ++v) { tu += uni[std::size_t(v)]; tb += row[v]; }
                h_uni -= std::log(uni[std::size_t(seq[i])] / tu);
                h_big -= std::log(row[seq[i]] / tb);
                ++n;
            }
        std::cout << std::fixed << std::setprecision(4)
                  << "  baselines on the held-out tokens (nats per token): unigram "
                  << h_uni / double(n) << ", per-field Markov " << h_big / double(n) << "\n";
    }

    Model model;
    CrossEntropy<float> ce;
    Adam<float> opt(3e-4f);
    std::mt19937 rng(7);
    std::size_t total = 0;
    for (const auto& s : train) total += s.size();
    const std::size_t steps = std::max<std::size_t>(200, 4 * total / BLOCK);

    std::cout << "  vocabulary " << VOCAB << " tokens;  model: " << N_LAYERS << " blocks, d_model "
              << D_MODEL << ", " << model.parameter_count() << " parameters\n"
              << "  window " << BLOCK << " tokens (" << BLOCK / PER_ATOM << " atoms), "
              << steps << " steps per epoch\n\n"
              << "epoch   train    held-out   time\n-----   ------   --------   -----\n";

    auto held_loss = [&]() {
        double sum = 0.0; std::size_t count = 0;
        for (const auto& seq : held)
            for (std::size_t off = 0; off + BLOCK + 1 <= seq.size(); off += BLOCK) {
                std::vector<int> in (seq.begin() + long(off), seq.begin() + long(off + BLOCK));
                std::vector<int> tgt(seq.begin() + long(off + 1), seq.begin() + long(off + BLOCK + 1));
                sum += ce.forward(model.forward(in), one_hot_seq(tgt));
                ++count;
            }
        return count ? sum / double(count) : 0.0;
    };

    const auto t0 = clock::now();
    double best = 1e9;
    for (int ep = 1; ep <= epochs; ++ep) {
        double sum = 0.0;
        for (std::size_t step = 0; step < steps; ++step) {
            const auto& seq = train[std::uniform_int_distribution<std::size_t>(0, train.size() - 1)(rng)];
            if (seq.size() < BLOCK + 2) continue;
            // Windows start at a multiple of 5, so that position mod 5 tells
            // the model which field it is writing (the lesson of the chorales).
            const std::size_t max_k = (seq.size() - BLOCK - 1) / PER_ATOM;
            std::size_t off = 0;
            if (std::uniform_int_distribution<int>(0, 19)(rng) != 0)
                off = PER_ATOM * std::uniform_int_distribution<std::size_t>(0, max_k)(rng);
            std::vector<int> in (seq.begin() + long(off), seq.begin() + long(off + BLOCK));
            std::vector<int> tgt(seq.begin() + long(off + 1), seq.begin() + long(off + BLOCK + 1));
            sum += ce.forward(model.forward(in), one_hot_seq(tgt));
            model.backward(ce.backward());
            model.update(opt);
        }
        const double h = held_loss();
        const double secs = std::chrono::duration<double>(clock::now() - t0).count();
        // Keep the weights with the lowest held-out loss: with a few sounds
        // the model soon starts memorising, and the training loss keeps
        // falling while the held-out loss turns up.
        const bool is_best = h < best;
        if (is_best) { best = h; model.save(prefix + ".atm"); }
        std::cout << std::setw(5) << ep << "   " << std::fixed << std::setprecision(4)
                  << sum / double(steps) << "   " << h << (is_best ? " *" : "  ") << "   "
                  << std::setprecision(0) << secs << "s\n" << std::flush;
        model.save(prefix + "_last.atm");
    }
    {
        std::ofstream meta(prefix + ".txt");
        meta << sample_rate << "\n";
        for (std::size_t N : sizes) meta << N << " ";
        meta << "\n";
        // The training token streams, for the "how much is copied" check.
        for (const auto& s : train) { for (int t : s) meta << t << " "; meta << "\n"; }
    }
    std::cout << "\nsaved " << prefix << ".atm (lowest held-out loss, marked *) and "
              << prefix << "_last.atm\n";
    return 0;
}

int cmd_generate(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: mp_generate generate <prefix> <out.wav> [seconds] [temperature] "
                     "[seed] [prime.atoms] [prime_seconds] [weights]\n";
        return 1;
    }
    const std::string prefix = argv[2];
    const double seconds   = (argc > 4) ? std::atof(argv[4]) : 10.0;
    const float  temp      = (argc > 5) ? float(std::atof(argv[5])) : 1.0f;
    const unsigned seed    = (argc > 6) ? unsigned(std::atoi(argv[6])) : 1u;
    const std::string prime = (argc > 7) ? argv[7] : "";
    const double prime_secs = (argc > 8) ? std::atof(argv[8]) : 1.0;
    const std::string weights = (argc > 9) ? argv[9] : prefix + ".atm";

    double sample_rate = 44100.0;
    std::vector<std::size_t> sizes;
    std::vector<std::vector<int>> train;
    if (!load_meta(prefix, sample_rate, sizes, train)) return 1;

    Model model;
    model.load(weights);
    model.set_training(false);

    std::vector<int> ctx = {TOK_BOS};
    long elapsed = 0;                                   // in DT units
    std::size_t prime_atoms = 0;
    if (!prime.empty()) {
        const AtomFile f = load_checked(prime);
        const std::vector<int> toks = atoms_to_tokens(select_atoms(f, 2000), f);
        for (std::size_t i = 1; i + PER_ATOM <= toks.size() - 1; i += PER_ATOM) {
            elapsed += dt_value(toks[i] - DT_BASE);
            if (double(elapsed) * DT_UNIT / sample_rate > prime_secs) break;
            for (int k = 0; k < PER_ATOM; ++k) ctx.push_back(toks[i + std::size_t(k)]);
            ++prime_atoms;
        }
    }

    std::mt19937 rng(seed);
    const long limit = long(seconds * sample_rate / DT_UNIT);
    while (elapsed < limit) {
        // The last BLOCK tokens, starting at a multiple of 5, as in training.
        std::size_t start = 0;
        if (ctx.size() > BLOCK)
            start = ((ctx.size() - BLOCK + PER_ATOM - 1) / PER_ATOM) * PER_ATOM;
        std::vector<int> window(ctx.begin() + long(start), ctx.end());
        const Matrix<float> probs = model.forward(window);
        const std::size_t col = probs.cols() - 1;

        // Only tokens of the field that comes next are allowed, so the
        // stream is always a well-formed list of atoms.
        int lo, hi;
        field_range(field_of(ctx.size()), lo, hi);
        std::vector<double> p(std::size_t(hi - lo));
        double sum = 0.0;
        for (int v = lo; v < hi; ++v) {
            p[std::size_t(v - lo)] = std::pow(double(std::max(1e-12f, probs(std::size_t(v), col))),
                                              1.0 / std::max(1e-3f, temp));
            sum += p[std::size_t(v - lo)];
        }
        double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
        int next = hi - 1;
        for (int v = lo; v < hi; ++v) {
            r -= p[std::size_t(v - lo)];
            if (r <= 0.0) { next = v; break; }
        }
        if (field_of(ctx.size()) == 0) elapsed += dt_value(next - DT_BASE);
        ctx.push_back(next);
    }
    while ((ctx.size() - 1) % PER_ATOM != 0) ctx.pop_back();      // drop a partial atom

    const std::vector<Atom> atoms =
        tokens_to_atoms(ctx, sizes, sample_rate, "token", seed);
    const std::size_t n_samples = std::size_t((seconds + 0.5) * sample_rate);
    save_wav(argv[3], render(atoms, sizes, n_samples), std::uint32_t(sample_rate));
    save_generated(argv[3], atoms, sizes, sample_rate, n_samples);

    // How much of this is a copy?  Count the generated runs of 8 atoms (32
    // tokens) that occur, token for token, in a training sound.
    const std::size_t RUN = 8 * PER_ATOM;
    std::set<std::vector<int>> seen;
    for (const auto& s : train)
        for (std::size_t i = 1; i + RUN <= s.size(); i += PER_ATOM)
            seen.insert(std::vector<int>(s.begin() + long(i), s.begin() + long(i + RUN)));
    std::size_t runs = 0, copied = 0;
    for (std::size_t i = 1 + prime_atoms * PER_ATOM; i + RUN <= ctx.size(); i += PER_ATOM) {
        ++runs;
        if (seen.count(std::vector<int>(ctx.begin() + long(i), ctx.begin() + long(i + RUN))))
            ++copied;
    }
    std::cout << "wrote " << argv[3] << "  (" << atoms.size() << " atoms, " << std::fixed
              << std::setprecision(1) << atoms.size() / seconds << " per second, "
              << prime_atoms << " from the prime;  " << std::setprecision(1)
              << (runs ? 100.0 * double(copied) / double(runs) : 0.0)
              << "% of 8-atom runs are verbatim copies of the training sounds)\n";
    return 0;
}

// The baseline: a first-order Markov chain over atoms.  Each of the five
// fields of an atom is drawn given only the same field of the previous
// atom, so the chain knows how often each value occurs and what tends to
// follow it one step back, and nothing else.  It is fitted on the same
// training token streams as the transformer.
int cmd_markov(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: mp_generate markov <prefix> <out.wav> [seconds] [seed]\n";
        return 1;
    }
    const std::string prefix = argv[2];
    const double seconds = (argc > 4) ? std::atof(argv[4]) : 10.0;
    const unsigned seed  = (argc > 5) ? unsigned(std::atoi(argv[5])) : 1u;
    double sample_rate = 44100.0;
    std::vector<std::size_t> sizes;
    std::vector<std::vector<int>> train;
    if (!load_meta(prefix, sample_rate, sizes, train)) return 1;

    std::vector<double> counts(std::size_t(VOCAB) * VOCAB, 0.0);
    for (const auto& seq : train)
        for (std::size_t i = 1 + PER_ATOM; i + 1 < seq.size(); ++i)
            counts[std::size_t(seq[i - PER_ATOM]) * VOCAB + std::size_t(seq[i])] += 1.0;

    std::mt19937 rng(seed);
    const auto& first = train[std::uniform_int_distribution<std::size_t>(0, train.size() - 1)(rng)];
    std::vector<int> ctx(first.begin(), first.begin() + 1 + PER_ATOM);     // BOS + one atom
    long elapsed = 0;
    const long limit = long(seconds * sample_rate / DT_UNIT);
    while (elapsed < limit) {
        const int prev = ctx[ctx.size() - PER_ATOM];
        int lo, hi;
        field_range(field_of(ctx.size()), lo, hi);
        const double* row = &counts[std::size_t(prev) * VOCAB];
        double total = 0.0;
        for (int v = lo; v < hi; ++v) total += row[v];
        int next = prev;                                   // unseen context: repeat
        if (total > 0.0) {
            double r = std::uniform_real_distribution<double>(0.0, total)(rng);
            for (int v = lo; v < hi; ++v) { r -= row[v]; if (r <= 0.0) { next = v; break; } }
        }
        if (field_of(ctx.size()) == 0) elapsed += dt_value(next - DT_BASE);
        ctx.push_back(next);
    }
    while ((ctx.size() - 1) % PER_ATOM != 0) ctx.pop_back();
    const std::vector<Atom> atoms = tokens_to_atoms(ctx, sizes, sample_rate, "token", seed);
    const std::size_t n_samples = std::size_t((seconds + 0.5) * sample_rate);
    save_wav(argv[3], render(atoms, sizes, n_samples), std::uint32_t(sample_rate));
    save_generated(argv[3], atoms, sizes, sample_rate, n_samples);
    std::cout << "wrote " << argv[3] << "  (per-field Markov chain, " << atoms.size()
              << " atoms)\n";
    return 0;
}

// Numbers that describe how the atoms of a sound are organised in time,
// so that generated sounds can be compared with real ones without
// listening.
//
//   continuity : of the atoms with a long window (the two longest), the
//                share that follow another long atom at the same frequency
//                (within one grid step) by less than their own length.  A
//                sustained partial is a chain of such atoms; in a sound
//                with no partials this is small.
//   clustering : of the atoms with a short window (the two shortest), the
//                share that have another short atom within 6 ms.  An attack
//                is a burst of them.
int cmd_stats(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: mp_generate stats <n_atoms> <a.atoms> [b.atoms ...]\n"
                     "       (n_atoms: use only the strongest n of each file; 0 = all)\n";
        return 1;
    }
    const std::size_t limit = std::atoi(argv[2]) > 0 ? std::size_t(std::atoi(argv[2]))
                                                     : std::size_t(2000000);
    std::cout << "                                   atoms/s   window shares (short -> long)   "
                 "continuity   clustering\n";
    for (int q = 3; q < argc; ++q) {
        const AtomFile f = load_checked(argv[q]);
        const std::vector<Atom> atoms = select_atoms(f, limit);
        if (atoms.empty()) continue;
        std::vector<long> centre(atoms.size());
        std::vector<int>  freq(atoms.size());
        std::vector<double> share(SCALE_N, 0.0);
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            centre[i] = centre_of(atoms[i], f.window_lengths);
            const double hz = double(atoms[i].bin) * f.sample_rate /
                              double(f.window_lengths[atoms[i].scale]);
            freq[i] = int(std::lround(STEPS_PER_OCTAVE * std::log2(std::max(hz, FREQ_MIN) / FREQ_MIN)));
            share[atoms[i].scale] += 1.0 / double(atoms.size());
        }
        std::size_t n_long = 0, cont = 0, n_short = 0, clus = 0;
        for (std::size_t i = 0; i < atoms.size(); ++i) {
            const bool is_long = atoms[i].scale >= SCALE_N / 2;
            if (is_long) {
                ++n_long;
                const long reach = long(f.window_lengths[atoms[i].scale]);
                for (std::size_t j = i; j-- > 0;) {
                    if (centre[i] - centre[j] >= reach) break;
                    if (centre[j] == centre[i]) continue;
                    if (atoms[j].scale >= SCALE_N / 2 && std::abs(freq[j] - freq[i]) <= 1) {
                        ++cont; break;
                    }
                }
            } else {
                ++n_short;
                bool found = false;
                for (std::size_t j = i; j-- > 0 && !found;) {
                    if (centre[i] - centre[j] > 256) break;
                    if (atoms[j].scale < SCALE_N / 2) found = true;
                }
                for (std::size_t j = i + 1; j < atoms.size() && !found; ++j) {
                    if (centre[j] - centre[i] > 256) break;
                    if (atoms[j].scale < SCALE_N / 2) found = true;
                }
                if (found) ++clus;
            }
        }
        const double dur = double(centre.back() - centre.front()) / f.sample_rate;
        std::string name = argv[q];
        const std::size_t slash = name.rfind('/');
        if (slash != std::string::npos) name = name.substr(slash + 1);
        if (name.size() > 32) name = name.substr(0, 32);
        std::cout << std::left << std::setw(34) << name << std::right << std::fixed
                  << std::setw(8) << std::setprecision(0) << atoms.size() / std::max(dur, 1e-9)
                  << "   ";
        for (double v : share) std::cout << std::setw(6) << std::setprecision(0) << 100.0 * v << "%";
        std::cout << "       " << std::setw(8) << std::setprecision(0)
                  << (n_long ? 100.0 * cont / n_long : 0.0) << "%   " << std::setw(8)
                  << (n_short ? 100.0 * clus / n_short : 0.0) << "%\n";
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = (argc > 1) ? argv[1] : "";
    try {
        if (cmd == "roundtrip") return cmd_roundtrip(argc, argv);
        if (cmd == "train")     return cmd_train(argc, argv);
        if (cmd == "generate")  return cmd_generate(argc, argv);
        if (cmd == "markov")    return cmd_markov(argc, argv);
        if (cmd == "stats")     return cmd_stats(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage:\n"
                 "  mp_generate roundtrip <in.atoms> <out.wav> [n_atoms] [token|original|locked|random]\n"
                 "  mp_generate train     <prefix> <epochs> <n_atoms> <a.atoms> [b.atoms ...]\n"
                 "  mp_generate generate  <prefix> <out.wav> [seconds] [temperature] [seed]\n"
                 "                        [prime.atoms] [prime_seconds] [weights]\n"
                 "  mp_generate markov    <prefix> <out.wav> [seconds] [seed]\n"
                 "  mp_generate stats     <n_atoms> <a.atoms> [b.atoms ...]\n";
    return 1;
}
