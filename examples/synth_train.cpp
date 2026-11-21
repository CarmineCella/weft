// synth_train / synth_train.cpp
//
// Train a network to PLAY a harmonic-plus-noise synthesiser.
//
// synth_fit.cpp fitted the synthesiser's controls to one note at a time.
// Here a single network learns to produce those controls for any note of
// any instrument in the training set.  It never outputs audio.  For every
// control frame it reads
//
//     pitch            log2(f0 / 440), scaled
//     loudness         the level of the sound around this frame, in dB,
//                      at nine time offsets (so the network can tell an
//                      attack from a decay at the same level)
//     instrument       a learned vector, one per instrument
//
// and outputs
//
//     one overall amplitude,
//     how that amplitude is shared among the harmonics,
//     the gain of each noise band.
//
// Those go into HarmonicSynth + FilteredNoise, the result is compared to
// the recording with SpectralLoss, and the gradient flows back through the
// synthesiser into the network.  The synthesiser has no parameters of its
// own: everything learned lives in the network and the instrument vectors.
//
// Because the instrument is just a vector, it can be changed at will after
// training: swap it (one instrument's pitch and dynamics, another's
// timbre), or interpolate between two (a timbre that does not exist).
//
// Usage:
//   synth_train train <wav_dir> <model_prefix> [epochs] [batch]
//   synth_train resynth <model_prefix> <in.wav> <note> <instrument> <out.wav>
//       pitch and loudness from in.wav, timbre of <instrument>
//   synth_train note <model_prefix> <instrA> <instrB> <note> <seconds> <shape> <out.wav>
//       a new note; the timbre glides from instrA to instrB (give the same
//       name twice for a plain note).  shape: flat | cresc | dim | swell |
//       vibrato | gliss
//   synth_train phrase <model_prefix> <instrument> <out.wav>
//       a short built-in melody
//   synth_train list <model_prefix>
//       instruments known to the model and their typical loudness

#include "Adam.h"
#include "Dense.h"
#include "Embedding.h"
#include "FilteredNoise.h"
#include "HarmonicSynth.h"
#include "Matrix.h"
#include "Network.h"
#include "Pitch.h"
#include "ReLU.h"
#include "SpectralLoss.h"
#include "Wav.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace weft;
namespace fs = std::filesystem;

namespace {

// ---- fixed sizes ---------------------------------------------------------
constexpr std::size_t HOP        = 512;    // samples per control frame
constexpr std::size_t K_HARM     = 100;    // harmonics
constexpr std::size_t N_BANDS    = 65;     // noise bands
constexpr std::size_t EMB_DIM    = 16;     // instrument vector size
constexpr std::size_t HIDDEN     = 256;
constexpr std::size_t CROP       = 128;    // control frames per training crop
constexpr std::size_t LOUD_WIN   = 4096;   // loudness analysis window
const int LOUD_OFFSETS[9] = {-8, -4, -2, -1, 0, 1, 2, 4, 8};
constexpr std::size_t N_FEAT     = 1 + 9;              // pitch + loudness context
constexpr std::size_t N_IN       = N_FEAT + EMB_DIM;
constexpr std::size_t N_OUT      = 1 + K_HARM + N_BANDS;

// Output activation (see synth_fit.cpp): positive, at most 2, roughly
// exponential for small values.
const float LN10 = 2.302585093f;
inline float act(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return 2.0f * std::pow(s, LN10) + 1e-7f;
}
inline float act_deriv(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return 2.0f * LN10 * std::pow(s, LN10) * (1.0f - s);
}
// Offsets so that an untrained network (outputs near 0) starts quiet.
constexpr float AMP_BIAS   = -3.0f;
constexpr float NOISE_BIAS = -5.0f;

// Loudness of the sound around each control frame: RMS in dB, mapped so
// that -80 dB -> 0 and 0 dB -> 1.
std::vector<float> loudness_curve(const std::vector<float>& x, std::size_t n_frames) {
    std::vector<float> out(n_frames);
    for (std::size_t t = 0; t < n_frames; ++t) {
        const long start = long(t * HOP) - long(LOUD_WIN / 2);
        double e = 0.0;
        for (std::size_t i = 0; i < LOUD_WIN; ++i) {
            const long idx = start + long(i);
            if (idx < 0 || idx >= long(x.size())) continue;
            e += double(x[std::size_t(idx)]) * x[std::size_t(idx)];
        }
        const double db = 10.0 * std::log10(e / double(LOUD_WIN) + 1e-10);
        out[t] = static_cast<float>(std::min(1.0, std::max(0.0, (db + 80.0) / 80.0)));
    }
    return out;
}

inline float pitch_feature(float f0) { return std::log2(f0 / 440.0f) / 4.0f; }

// Feature columns for frames [t0, t0 + n) of a pitch and loudness curve.
// Rows 0..N_FEAT-1; the instrument vector is appended by the caller.
void fill_features(Matrix<float>& X, std::size_t col0,
                   const std::vector<float>& f0, const std::vector<float>& loud,
                   std::size_t t0, std::size_t n)
{
    const long T = long(loud.size());
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t t = t0 + j;
        X(0, col0 + j) = pitch_feature(f0[t]);
        for (int q = 0; q < 9; ++q) {
            long u = long(t) + LOUD_OFFSETS[q];
            u = std::min(T - 1, std::max(0L, u));
            X(1 + q, col0 + j) = loud[std::size_t(u)];
        }
    }
}

// ---- the head: network outputs -> synthesiser controls -------------------
//
// For one column o of the network output and the frame's f0:
//   amp   = act(o[0] + AMP_BIAS)
//   h_k   = act(o[1+k]) for harmonics below Nyquist, 0 above
//   A_k   = amp * h_k / sum_j h_j          (harmonic amplitudes)
//   M_b   = act(o[1+K+b] + NOISE_BIAS)     (noise band gains)
// Splitting the harmonics into "how loud overall" and "how it is shared"
// lets the network learn timbre (the shares) separately from dynamics.
struct Head {
    static void forward(const Matrix<float>& O, std::size_t col0,
                        const std::vector<float>& f0, std::size_t t0, std::size_t n,
                        float sample_rate, Matrix<float>& A, Matrix<float>& M)
    {
        const float nyq = 0.5f * sample_rate;
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t c = col0 + j;
            const float amp = act(O(0, c) + AMP_BIAS);
            float sum = 0.0f;
            for (std::size_t k = 0; k < K_HARM; ++k) {
                const bool ok = float(k + 1) * f0[t0 + j] < nyq;
                const float h = ok ? act(O(1 + k, c)) : 0.0f;
                A(k, j) = h;
                sum += h;
            }
            const float inv = (sum > 0.0f) ? amp / sum : 0.0f;
            for (std::size_t k = 0; k < K_HARM; ++k) A(k, j) *= inv;
            for (std::size_t b = 0; b < N_BANDS; ++b)
                M(b, j) = act(O(1 + K_HARM + b, c) + NOISE_BIAS);
        }
    }

    // dA, dM -> dO (written into columns col0..col0+n of dO).
    static void backward(const Matrix<float>& O, std::size_t col0,
                         const std::vector<float>& f0, std::size_t t0, std::size_t n,
                         float sample_rate, const Matrix<float>& A,
                         const Matrix<float>& dA, const Matrix<float>& dM,
                         Matrix<float>& dO)
    {
        const float nyq = 0.5f * sample_rate;
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t c = col0 + j;
            const float amp = act(O(0, c) + AMP_BIAS);
            // share_k = A_k / amp;  sum_h = sum of raw h.
            float sum_h = 0.0f, g_amp = 0.0f, dot = 0.0f;
            for (std::size_t k = 0; k < K_HARM; ++k) {
                if (!(float(k + 1) * f0[t0 + j] < nyq)) continue;
                sum_h += act(O(1 + k, c));
                const float share = A(k, j) / amp;
                g_amp += dA(k, j) * share;          // dL/d amp
                dot   += dA(k, j) * amp * share;    // sum_j dL/dshare_j * share_j
            }
            dO(0, c) = g_amp * act_deriv(O(0, c) + AMP_BIAS);
            for (std::size_t k = 0; k < K_HARM; ++k) {
                if (!(float(k + 1) * f0[t0 + j] < nyq) || sum_h <= 0.0f) {
                    dO(1 + k, c) = 0.0f;
                    continue;
                }
                // share_k = h_k / sum_h  =>  dL/dh_k = (dL/dshare_k - dot) / sum_h
                const float dh = (dA(k, j) * amp - dot) / sum_h;
                dO(1 + k, c) = dh * act_deriv(O(1 + k, c));
            }
            for (std::size_t b = 0; b < N_BANDS; ++b)
                dO(1 + K_HARM + b, c) =
                    dM(b, j) * act_deriv(O(1 + K_HARM + b, c) + NOISE_BIAS);
        }
    }
};

// FFT sizes for the loss, by the rule found in synth_fit.cpp: at least 2.5
// periods of the note, the five smallest such sizes, capped at 8192.
std::vector<std::size_t> loss_sizes(float sample_rate, float f0) {
    std::vector<std::size_t> sizes;
    const float period = sample_rate / f0;
    for (std::size_t N = 64; N <= 8192 && sizes.size() < 5; N *= 2)
        if (float(N) >= 2.5f * period) sizes.push_back(N);
    if (sizes.empty()) sizes.push_back(8192);
    return sizes;
}

// ---- the model -----------------------------------------------------------
struct Model {
    Network<float>   net;
    Embedding<float> emb;
    std::vector<std::string> instruments;
    std::vector<float>       typical_loud;   // median sustain loudness, per instrument
    float sample_rate = 44100.0f;

    explicit Model(std::size_t n_instruments)
        : emb(n_instruments, EMB_DIM, 7)
    {
        net.add<Dense>(N_IN, HIDDEN);
        net.add<ReLU>();
        net.add<Dense>(HIDDEN, HIDDEN);
        net.add<ReLU>();
        net.add<Dense>(HIDDEN, HIDDEN);
        net.add<ReLU>();
        net.add<Dense>(HIDDEN, N_OUT);
    }

    void save(const std::string& prefix) const {
        net.save(prefix + ".net");
        std::ofstream e(prefix + ".emb", std::ios::binary);
        emb.save_params(e);
        std::ofstream t(prefix + ".txt");
        t << sample_rate << "\n" << instruments.size() << "\n";
        for (std::size_t i = 0; i < instruments.size(); ++i)
            t << instruments[i] << " " << typical_loud[i] << "\n";
    }

    static Model load(const std::string& prefix) {
        std::ifstream t(prefix + ".txt");
        if (!t) throw std::runtime_error("cannot open " + prefix + ".txt");
        float sr; std::size_t n;
        t >> sr >> n;
        Model m(n);
        m.sample_rate = sr;
        m.instruments.resize(n);
        m.typical_loud.resize(n);
        for (std::size_t i = 0; i < n; ++i) t >> m.instruments[i] >> m.typical_loud[i];
        m.net.load(prefix + ".net");
        std::ifstream e(prefix + ".emb", std::ios::binary);
        if (!e) throw std::runtime_error("cannot open " + prefix + ".emb");
        m.emb.load_params(e);
        return m;
    }

    int instrument_id(const std::string& name) const {
        for (std::size_t i = 0; i < instruments.size(); ++i)
            if (instruments[i] == name) return int(i);
        throw std::runtime_error("unknown instrument '" + name +
                                 "' (run: synth_train list <model_prefix>)");
    }

    // Render audio from a pitch curve, a loudness curve and, per frame, a
    // mix between two instrument vectors (mix = 0 -> a, 1 -> b).
    std::vector<float> render(const std::vector<float>& f0,
                              const std::vector<float>& loud,
                              int inst_a, int inst_b,
                              const std::vector<float>& mix)
    {
        const std::size_t T = f0.size();
        const std::size_t n = (T - 1) * HOP;
        Matrix<float> X(N_IN, T);
        fill_features(X, 0, f0, loud, 0, T);
        const Matrix<float>& W = emb.W();
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t d = 0; d < EMB_DIM; ++d)
                X(N_FEAT + d, t) = (1.0f - mix[t]) * W(d, std::size_t(inst_a))
                                 + mix[t] * W(d, std::size_t(inst_b));
        const Matrix<float> O = net.forward(X);
        Matrix<float> A(K_HARM, T), M(N_BANDS, T);
        Head::forward(O, 0, f0, 0, T, sample_rate, A, M);
        HarmonicSynth<float> synth(sample_rate, HOP);
        FilteredNoise<float> noise(HOP, N_BANDS);
        auto y = synth.forward(f0, A, n);
        const auto z = noise.forward(M, n);
        for (std::size_t i = 0; i < n; ++i) y[i] += z[i];
        return y;
    }
};

// ---- data ----------------------------------------------------------------
struct Note {
    std::string        name;
    int                inst = 0;
    float              nominal = 440.0f;
    std::vector<float> audio, f0, loud;
    bool               held_out = false;
};

// "Vc-ord-A#3-mf-2c.wav" -> instrument "Vc", note "A#3".
bool parse_name(const std::string& stem, std::string& inst, std::string& note) {
    std::vector<std::string> parts;
    std::stringstream ss(stem);
    std::string item;
    while (std::getline(ss, item, '-')) parts.push_back(item);
    if (parts.size() < 3) return false;
    inst = parts[0];
    note = parts[2];
    return !note.empty() && note[0] >= 'A' && note[0] <= 'G';
}

int cmd_train(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: synth_train train <wav_dir> <model_prefix> [epochs] [batch]\n";
        return 1;
    }
    using clock = std::chrono::steady_clock;
    const std::string dir    = argv[2];
    const std::string prefix = argv[3];
    const int epochs         = (argc > 4) ? std::atoi(argv[4]) : 40;
    const std::size_t hw     = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t batch  = (argc > 5) ? std::size_t(std::atoi(argv[5])) : hw;

    // ---- scan ----
    std::vector<fs::path> paths;
    for (const auto& e : fs::recursive_directory_iterator(dir))
        if (e.path().extension() == ".wav") paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());

    std::map<std::string, int> inst_id;
    std::vector<Note> notes(paths.size());
    std::vector<bool> ok(paths.size(), false);
    for (std::size_t i = 0; i < paths.size(); ++i) {
        std::string inst, note;
        if (!parse_name(paths[i].stem().string(), inst, note)) continue;
        if (!inst_id.count(inst)) { const int id = int(inst_id.size()); inst_id[inst] = id; }
        notes[i].name    = paths[i].stem().string();
        notes[i].inst    = inst_id[inst];
        notes[i].nominal = static_cast<float>(note_to_hz(note));
        ok[i] = true;
    }

    std::cout << "weft :: synth_train\n  data: " << dir << "  (" << paths.size()
              << " files, " << inst_id.size() << " instruments)\n"
              << "  loading audio, refining pitch, measuring loudness..." << std::flush;
    const auto t_load = clock::now();
    float sample_rate = 44100.0f;
    {
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        for (std::size_t w = 0; w < hw; ++w)
            pool.emplace_back([&] {
                for (;;) {
                    const std::size_t i = next++;
                    if (i >= paths.size()) return;
                    if (!ok[i]) continue;
                    try {
                        WavData wav = load_wav(paths[i].string());
                        Note& n = notes[i];
                        n.audio = std::move(wav.samples);
                        const std::size_t T = n.audio.size() / HOP + 1;
                        if (T < CROP + 2) { ok[i] = false; continue; }
                        n.f0   = refine_f0<float>(n.audio, float(wav.sample_rate),
                                                  n.nominal, HOP, T);
                        n.loud = loudness_curve(n.audio, T);
                    } catch (const std::exception&) { ok[i] = false; }
                }
            });
        for (auto& th : pool) th.join();
    }
    {
        std::vector<Note> kept;
        for (std::size_t i = 0; i < notes.size(); ++i)
            if (ok[i]) kept.push_back(std::move(notes[i]));
        notes = std::move(kept);
    }
    // Every 20th note is held out: never trained on, used for the
    // validation loss and for the "notes the model never heard" examples.
    std::vector<std::size_t> train_idx, val_idx;
    for (std::size_t i = 0; i < notes.size(); ++i) {
        notes[i].held_out = (i % 20 == 10);
        (notes[i].held_out ? val_idx : train_idx).push_back(i);
    }
    std::cout << " done ("
              << std::chrono::duration<double>(clock::now() - t_load).count()
              << "s)\n  notes: " << train_idx.size() << " training, "
              << val_idx.size() << " held out\n";

    Model model(inst_id.size());
    model.sample_rate = sample_rate;
    model.instruments.resize(inst_id.size());
    for (const auto& kv : inst_id) model.instruments[std::size_t(kv.second)] = kv.first;
    // Typical sustain loudness per instrument: the median, over the
    // instrument's notes, of each note's 75th-percentile loudness.  Used
    // only by the generators.
    model.typical_loud.assign(inst_id.size(), 0.6f);
    for (std::size_t q = 0; q < inst_id.size(); ++q) {
        std::vector<float> v;
        for (const Note& n : notes)
            if (n.inst == int(q)) {
                std::vector<float> l = n.loud;
                std::sort(l.begin(), l.end());
                v.push_back(l[l.size() * 3 / 4]);
            }
        if (!v.empty()) {
            std::sort(v.begin(), v.end());
            model.typical_loud[q] = v[v.size() / 2];
        }
    }
    {
        std::ofstream held(prefix + "_held_out.txt");
        for (std::size_t i : val_idx) held << notes[i].name << "\n";
    }

    std::cout << "  network:\n" << model.net.summary() << "\n"
              << "  inputs: pitch + 9 loudness values + " << EMB_DIM
              << "-D instrument vector;  outputs: 1 amplitude + " << K_HARM
              << " harmonic shares + " << N_BANDS << " noise bands\n"
              << "  batch " << batch << " notes, crop " << CROP << " frames ("
              << std::fixed << std::setprecision(2) << (CROP - 1) * HOP / sample_rate
              << " s)\n\n";

    Adam<float> opt(1e-3f);
    std::mt19937 rng(1234);

    // One pass of the network + synthesiser + loss over a set of crops.
    // Returns the mean loss; if `learn`, also updates the parameters.
    auto run_batch = [&](const std::vector<std::size_t>& idx,
                         const std::vector<std::size_t>& start, bool learn,
                         unsigned noise_seed) {
        const std::size_t B = idx.size();
        Matrix<float> X(N_IN, B * CROP);
        std::vector<int> tokens(B * CROP);
        for (std::size_t b = 0; b < B; ++b) {
            const Note& n = notes[idx[b]];
            fill_features(X, b * CROP, n.f0, n.loud, start[b], CROP);
            for (std::size_t j = 0; j < CROP; ++j) tokens[b * CROP + j] = n.inst;
        }
        const Matrix<float> E = model.emb.forward(tokens);        // (EMB_DIM x B*CROP)
        for (std::size_t c = 0; c < B * CROP; ++c)
            for (std::size_t d = 0; d < EMB_DIM; ++d) X(N_FEAT + d, c) = E(d, c);

        const Matrix<float> O = model.net.forward(X);
        Matrix<float> dO(N_OUT, B * CROP, 0.0f);
        std::vector<float> losses(B, 0.0f);

        // The synthesiser and the loss run once per note, in parallel.
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        const std::size_t workers = std::min(hw, B);
        const std::size_t n_samp = (CROP - 1) * HOP;
        for (std::size_t w = 0; w < workers; ++w)
            pool.emplace_back([&] {
                for (;;) {
                    const std::size_t b = next++;
                    if (b >= B) return;
                    const Note& n = notes[idx[b]];
                    std::vector<float> f0(n.f0.begin() + long(start[b]),
                                          n.f0.begin() + long(start[b] + CROP));
                    Matrix<float> A(K_HARM, CROP), M(N_BANDS, CROP);
                    Head::forward(O, b * CROP, f0, 0, CROP, sample_rate, A, M);

                    HarmonicSynth<float> synth(sample_rate, HOP);
                    FilteredNoise<float> noise(HOP, N_BANDS, noise_seed + unsigned(b));
                    auto y = synth.forward(f0, A, n_samp);
                    const auto z = noise.forward(M, n_samp);
                    for (std::size_t i = 0; i < n_samp; ++i) y[i] += z[i];

                    std::vector<float> target(
                        n.audio.begin() + long(start[b] * HOP),
                        n.audio.begin() + long(start[b] * HOP + n_samp));
                    SpectralLoss<float> loss(loss_sizes(sample_rate, n.nominal));
                    loss.set_target(target);
                    losses[b] = loss.forward(y);
                    if (!learn) continue;

                    auto dy = loss.backward();
                    const float scale = 1.0f / float(B);
                    for (float& v : dy) v *= scale;
                    const Matrix<float> dA = synth.backward(dy);
                    const Matrix<float> dM = noise.backward(dy);
                    Head::backward(O, b * CROP, f0, 0, CROP, sample_rate, A, dA, dM, dO);
                }
            });
        for (auto& th : pool) th.join();

        float mean = 0.0f;
        for (float l : losses) mean += l;
        mean /= float(B);

        if (learn) {
            const Matrix<float> dX = model.net.backward(dO);
            Matrix<float> dE(EMB_DIM, B * CROP);
            for (std::size_t c = 0; c < B * CROP; ++c)
                for (std::size_t d = 0; d < EMB_DIM; ++d) dE(d, c) = dX(N_FEAT + d, c);
            model.emb.backward(dE);
            model.net.update(opt);
            model.emb.update(opt);
        }
        return mean;
    };

    auto random_start = [&](const Note& n) {
        const std::size_t max_start = n.f0.size() - CROP - 1;
        return std::uniform_int_distribution<std::size_t>(0, max_start)(rng);
    };

    std::cout << "epoch   train    held-out   time\n-----   ------   --------   -----\n";
    const auto t0 = clock::now();
    for (int ep = 1; ep <= epochs; ++ep) {
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        double sum = 0.0; std::size_t count = 0;
        for (std::size_t s = 0; s + batch <= train_idx.size(); s += batch) {
            std::vector<std::size_t> idx(train_idx.begin() + long(s),
                                         train_idx.begin() + long(s + batch));
            std::vector<std::size_t> start(batch);
            for (std::size_t b = 0; b < batch; ++b) start[b] = random_start(notes[idx[b]]);
            sum += run_batch(idx, start, true, unsigned(rng()));
            ++count;
        }
        // Held-out loss on fixed crops (the first CROP frames after frame 4),
        // so the number is comparable from epoch to epoch.
        double vsum = 0.0; std::size_t vcount = 0;
        for (std::size_t s = 0; s < val_idx.size(); s += batch) {
            const std::size_t e = std::min(val_idx.size(), s + batch);
            std::vector<std::size_t> idx(val_idx.begin() + long(s), val_idx.begin() + long(e));
            std::vector<std::size_t> start(idx.size(), 4);
            vsum += run_batch(idx, start, false, 99u) * double(idx.size());
            vcount += idx.size();
        }
        const double secs = std::chrono::duration<double>(clock::now() - t0).count();
        std::cout << std::setw(5) << ep << "   " << std::setprecision(4)
                  << sum / double(count) << "   " << vsum / double(vcount) << "     "
                  << std::setprecision(0) << secs << "s\n" << std::flush;
        model.save(prefix);
    }
    std::cout << "\nsaved " << prefix << ".net / .emb / .txt   (held-out note names in "
              << prefix << "_held_out.txt)\n";
    return 0;
}

// ---- generators ----------------------------------------------------------

int cmd_list(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: synth_train list <model_prefix>\n"; return 1; }
    Model m = Model::load(argv[2]);
    std::cout << "instrument   typical loudness (0..1)\n";
    for (std::size_t i = 0; i < m.instruments.size(); ++i)
        std::cout << "  " << std::left << std::setw(10) << m.instruments[i]
                  << std::fixed << std::setprecision(3) << m.typical_loud[i] << "\n";
    return 0;
}

int cmd_resynth(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "usage: synth_train resynth <model_prefix> <in.wav> <note> "
                     "<instrument> <out.wav>\n";
        return 1;
    }
    Model m = Model::load(argv[2]);
    const WavData wav = load_wav(argv[3]);
    const float nominal = static_cast<float>(note_to_hz(argv[4]));
    const int inst = m.instrument_id(argv[5]);
    const std::size_t T = wav.samples.size() / HOP + 1;
    const auto f0   = refine_f0<float>(wav.samples, float(wav.sample_rate), nominal, HOP, T);
    const auto loud = loudness_curve(wav.samples, T);
    const std::vector<float> mix(T, 0.0f);
    const auto y = m.render(f0, loud, inst, inst, mix);
    save_wav(argv[6], y, wav.sample_rate);
    std::cout << "wrote " << argv[6] << "\n";
    return 0;
}

// A loudness curve for an invented note: attack, sustain at `level`
// (optionally shaped), release.
std::vector<float> envelope(std::size_t T, float level, const std::string& shape,
                            float sample_rate)
{
    const float fps = sample_rate / float(HOP);
    const std::size_t att = std::size_t(0.08f * fps), rel = std::size_t(0.25f * fps);
    std::vector<float> l(T);
    for (std::size_t t = 0; t < T; ++t) {
        const float x = float(t) / float(T - 1);
        float sustain = level;
        if (shape == "cresc")      sustain = level - 0.30f + 0.38f * x;
        else if (shape == "dim")   sustain = level + 0.08f - 0.38f * x;
        else if (shape == "swell") sustain = level - 0.25f + 0.33f * std::sin(3.14159265f * x);
        float g = 1.0f;
        if (t < att) g = float(t) / float(att);
        if (t + rel >= T) g = std::min(g, float(T - 1 - t) / float(rel));
        l[t] = std::max(0.0f, sustain * (0.25f + 0.75f * g) * (g > 0.0f ? 1.0f : 0.0f));
    }
    return l;
}

int cmd_note(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "usage: synth_train note <model_prefix> <instrA> <instrB> <note> "
                     "<seconds> <flat|cresc|dim|swell|vibrato|gliss> <out.wav>\n";
        return 1;
    }
    Model m = Model::load(argv[2]);
    const int a = m.instrument_id(argv[3]), b = m.instrument_id(argv[4]);
    const float hz = static_cast<float>(note_to_hz(argv[5]));
    const float secs = std::stof(argv[6]);
    const std::string shape = argv[7];
    const float fps = m.sample_rate / float(HOP);
    const std::size_t T = std::size_t(secs * fps) + 1;

    const float level = 0.5f * (m.typical_loud[std::size_t(a)] + m.typical_loud[std::size_t(b)]);
    const auto loud = envelope(T, level, shape, m.sample_rate);
    std::vector<float> f0(T, hz), mix(T, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float x = float(t) / float(T - 1);
        if (shape == "vibrato") {
            // 5.5 Hz vibrato growing to +-35 cents.
            const float depth = 35.0f * std::min(1.0f, x * 2.0f);
            f0[t] = hz * std::pow(2.0f, depth / 1200.0f *
                                  std::sin(2.0f * 3.14159265f * 5.5f * float(t) / fps));
        } else if (shape == "gliss") {
            f0[t] = hz * std::pow(2.0f, x);                 // up one octave
        }
        // Timbre: hold A for the first fifth, B for the last fifth.
        mix[t] = (a == b) ? 0.0f : std::min(1.0f, std::max(0.0f, (x - 0.2f) / 0.6f));
    }
    const auto y = m.render(f0, loud, a, b, mix);
    save_wav(argv[8], y, static_cast<std::uint32_t>(m.sample_rate));
    std::cout << "wrote " << argv[8] << "\n";
    return 0;
}

int cmd_phrase(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: synth_train phrase <model_prefix> <instrument> <out.wav> "
                     "[octave_shift]\n";
        return 1;
    }
    Model m = Model::load(argv[2]);
    const int inst = m.instrument_id(argv[3]);
    const int shift = (argc > 5) ? std::atoi(argv[5]) : 0;
    // An eight-note figure: (note, beats).
    const char* names[8]  = {"D4", "F4", "A4", "D5", "C#5", "A4", "E4", "D4"};
    const float beats[8]  = {1, 1, 1, 2, 1, 1, 1, 3};
    const float fps = m.sample_rate / float(HOP);
    const float beat = 0.42f;
    const float level = m.typical_loud[std::size_t(inst)];

    std::vector<float> f0, loud;
    for (int i = 0; i < 8; ++i) {
        const std::size_t T = std::size_t(beats[i] * beat * fps);
        const float hz = static_cast<float>(note_to_hz(names[i])) * std::pow(2.0f, float(shift));
        // Slight phrase shape: louder towards the long note in the middle.
        const float lv = level - 0.06f + 0.06f * std::sin(3.14159265f * float(i) / 7.0f);
        const auto env = envelope(T, lv, "flat", m.sample_rate);
        for (std::size_t t = 0; t < T; ++t) { f0.push_back(hz); loud.push_back(env[t]); }
    }
    const std::vector<float> mix(f0.size(), 0.0f);
    const auto y = m.render(f0, loud, inst, inst, mix);
    save_wav(argv[4], y, static_cast<std::uint32_t>(m.sample_rate));
    std::cout << "wrote " << argv[4] << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    weft::set_seed(1);
    const std::string cmd = (argc > 1) ? argv[1] : "";
    try {
        if (cmd == "train")   return cmd_train(argc, argv);
        if (cmd == "list")    return cmd_list(argc, argv);
        if (cmd == "resynth") return cmd_resynth(argc, argv);
        if (cmd == "note")    return cmd_note(argc, argv);
        if (cmd == "phrase")  return cmd_phrase(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage:\n"
                 "  synth_train train   <wav_dir> <model_prefix> [epochs] [batch]\n"
                 "  synth_train list    <model_prefix>\n"
                 "  synth_train resynth <model_prefix> <in.wav> <note> <instrument> <out.wav>\n"
                 "  synth_train note    <model_prefix> <instrA> <instrB> <note> <seconds> "
                 "<shape> <out.wav>\n"
                 "  synth_train phrase  <model_prefix> <instrument> <out.wav> [octave_shift]\n";
    return 1;
}
