// synth_fit / synth_fit.cpp
//
// Fit a harmonic-plus-noise synthesiser to a recorded note by gradient
// descent -- with NO neural network.  The "parameters" being learned are
// the synthesiser's own controls: the amplitude of every harmonic and the
// gain of every noise band, at every control frame.
//
// This is the experiment to run before training a network to produce
// those controls.  It answers one question: how good can this synthesiser
// sound at all?  Whatever a network later predicts, it cannot do better
// than controls fitted directly to the target, so this is the ceiling.
//
// The loop is the one every example in the library uses, with the
// synthesiser in the place of the network:
//
//     controls -> HarmonicSynth + FilteredNoise -> audio
//     audio, target -> SpectralLoss -> dL/d(audio)
//     dL/d(audio) -> synth backward -> dL/d(controls) -> Adam
//
// Pitch is not learned.  It comes from the note name (e.g. "A4") and is
// refined frame by frame against the recording (see Pitch.h).
//
// Usage:
//   synth_fit fit    <in.wav> <note|Hz> <prefix> [iterations] [fft_sizes]
//                    [learning_rate] [adam_eps] [knot_spacing]
//       (the last four are for experiments; the defaults are chosen from
//       the pitch, see the comments in cmd_fit)
//       writes <prefix>_resynth.wav, <prefix>_harmonic.wav,
//              <prefix>_noise.wav and <prefix>.ctl (the fitted controls)
//   synth_fit morph  <a.ctl> <b.ctl> <out.wav>
//       glides from the controls of A to the controls of B
//   synth_fit blend  <a.ctl> <b.ctl> <out.wav> [mix]
//       a fixed mixture of the two control sets (mix = 0.5 by default)
//   synth_fit render <in.ctl> <out.wav> [semitones] [glide_semitones]
//       re-renders the controls at a transposed pitch, optionally gliding
//       by a further glide_semitones over the length of the note

#include "Adam.h"
#include "FilteredNoise.h"
#include "HarmonicSynth.h"
#include "Matrix.h"
#include "Pitch.h"
#include "SpectralLoss.h"
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

constexpr std::size_t HOP          = 512;   // samples between control frames
constexpr std::size_t MAX_HARMONICS = 100;
constexpr std::size_t NOISE_BANDS  = 65;

// Output activation for amplitudes and gains, the one DDSP uses:
//   a = 2 * sigmoid(x)^ln(10) + 1e-7
// Always positive, at most 2, and close to exponential in x for small
// values, so equal steps in x are roughly equal steps in decibels.
const float LOG10 = 2.302585093f;
inline float act(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return 2.0f * std::pow(s, LOG10) + 1e-7f;
}
inline float act_deriv(float x) {
    const float s = 1.0f / (1.0f + std::exp(-x));
    return 2.0f * LOG10 * std::pow(s, LOG10) * (1.0f - s);
}

// The fitted controls of one note.
struct Controls {
    float               sample_rate = 44100.0f;
    std::size_t         n_samples   = 0;
    std::vector<float>  f0;         // T frames
    Matrix<float>       A;          // (K x T) harmonic amplitudes
    Matrix<float>       M;          // (B x T) noise band gains
};

void write_u32(std::ofstream& o, std::uint32_t v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
std::uint32_t read_u32(std::ifstream& i) {
    std::uint32_t v = 0;
    i.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
void write_matrix(std::ofstream& o, const Matrix<float>& X) {
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) {
            const float v = X(i, j);
            o.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
}
void read_matrix(std::ifstream& in, Matrix<float>& X) {
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) {
            float v = 0.0f;
            in.read(reinterpret_cast<char*>(&v), sizeof(v));
            X(i, j) = v;
        }
}

void save_controls(const std::string& path, const Controls& c) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + path);
    o.write("WCTL", 4);
    write_u32(o, static_cast<std::uint32_t>(c.sample_rate));
    write_u32(o, static_cast<std::uint32_t>(c.n_samples));
    write_u32(o, static_cast<std::uint32_t>(c.f0.size()));
    write_u32(o, static_cast<std::uint32_t>(c.A.rows()));
    write_u32(o, static_cast<std::uint32_t>(c.M.rows()));
    o.write(reinterpret_cast<const char*>(c.f0.data()),
            static_cast<std::streamsize>(c.f0.size() * sizeof(float)));
    write_matrix(o, c.A);
    write_matrix(o, c.M);
}

Controls load_controls(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "WCTL")
        throw std::runtime_error("not a controls file: " + path);
    Controls c;
    c.sample_rate = static_cast<float>(read_u32(in));
    c.n_samples   = read_u32(in);
    const std::size_t T = read_u32(in), K = read_u32(in), B = read_u32(in);
    c.f0.resize(T);
    in.read(reinterpret_cast<char*>(c.f0.data()),
            static_cast<std::streamsize>(T * sizeof(float)));
    c.A = Matrix<float>(K, T);
    c.M = Matrix<float>(B, T);
    read_matrix(in, c.A);
    read_matrix(in, c.M);
    if (!in) throw std::runtime_error("short read in " + path);
    return c;
}

// Controls -> audio.  Optionally returns the two components separately.
std::vector<float> render(const Controls& c, std::vector<float>* harmonic = nullptr,
                          std::vector<float>* noise_part = nullptr)
{
    HarmonicSynth<float> synth(c.sample_rate, HOP);
    FilteredNoise<float> noise(HOP, c.M.rows());
    auto h = synth.forward(c.f0, c.A, c.n_samples);
    auto z = noise.forward(c.M, c.n_samples);
    std::vector<float> y(c.n_samples);
    for (std::size_t i = 0; i < c.n_samples; ++i) y[i] = h[i] + z[i];
    if (harmonic)   *harmonic   = std::move(h);
    if (noise_part) *noise_part = std::move(z);
    return y;
}

// Mix two control sets.  mix(t) in [0, 1] gives the share of B at frame t.
// The result has the shorter of the two lengths and the larger harmonic
// count (the shorter set is padded with silent harmonics).
template <typename MixFn>
Controls combine(const Controls& a, const Controls& b, MixFn mix) {
    if (a.sample_rate != b.sample_rate)
        throw std::runtime_error("the two control sets have different sample rates");
    if (a.M.rows() != b.M.rows())
        throw std::runtime_error("the two control sets have different noise bands");
    Controls c;
    c.sample_rate = a.sample_rate;
    c.n_samples   = std::min(a.n_samples, b.n_samples);
    const std::size_t T = std::min(a.f0.size(), b.f0.size());
    const std::size_t K = std::max(a.A.rows(), b.A.rows());
    const std::size_t B = a.M.rows();
    c.f0.resize(T);
    c.A = Matrix<float>(K, T, 0.0f);
    c.M = Matrix<float>(B, T, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float m = mix(t, T);
        // Pitch is mixed on a logarithmic scale (a glide in semitones).
        c.f0[t] = std::exp((1.0f - m) * std::log(a.f0[t]) + m * std::log(b.f0[t]));
        for (std::size_t k = 0; k < K; ++k) {
            const float va = (k < a.A.rows()) ? a.A(k, t) : 0.0f;
            const float vb = (k < b.A.rows()) ? b.A(k, t) : 0.0f;
            c.A(k, t) = (1.0f - m) * va + m * vb;
        }
        for (std::size_t q = 0; q < B; ++q)
            c.M(q, t) = (1.0f - m) * a.M(q, t) + m * b.M(q, t);
    }
    return c;
}

int cmd_fit(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: synth_fit fit <in.wav> <note|Hz> <prefix> [iterations]\n";
        return 1;
    }
    using clock = std::chrono::steady_clock;
    const std::string in_path = argv[2];
    const std::string pitch   = argv[3];
    const std::string prefix  = argv[4];
    const int iterations      = (argc > 5) ? std::atoi(argv[5]) : 400;
    // Optional comma-separated FFT sizes for the loss, e.g. 4096,2048,1024.
    std::vector<std::size_t> fft_sizes;
    if (argc > 6) {
        std::string list = argv[6];
        std::size_t pos = 0;
        while (pos < list.size()) {
            const std::size_t comma = list.find(',', pos);
            const std::string item = list.substr(pos, comma - pos);
            fft_sizes.push_back(static_cast<std::size_t>(std::stoul(item)));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    const WavData wav = load_wav(in_path);
    const float sr = static_cast<float>(wav.sample_rate);
    const std::vector<float>& target = wav.samples;
    const std::size_t n = target.size();

    const bool is_number = !pitch.empty() &&
        (std::isdigit(static_cast<unsigned char>(pitch[0])) != 0);
    const float nominal = is_number ? std::stof(pitch)
                                    : static_cast<float>(note_to_hz(pitch));

    const std::size_t T = HarmonicSynth<float>::frames_for(n, HOP);
    const std::size_t K = std::min<std::size_t>(
        MAX_HARMONICS, static_cast<std::size_t>(0.5f * sr / nominal));

    std::cout << "weft :: synth_fit\n"
              << "  input:     " << in_path << "  (" << n << " samples @ "
              << wav.sample_rate << " Hz, " << std::fixed << std::setprecision(2)
              << n / sr << " s)\n"
              << "  pitch:     " << pitch << " = " << nominal << " Hz nominal\n";

    Controls c;
    c.sample_rate = sr;
    c.n_samples   = n;
    c.f0 = refine_f0<float>(target, sr, nominal, HOP, T);
    {
        std::vector<float> sorted = c.f0;
        std::sort(sorted.begin(), sorted.end());
        const float med = sorted[sorted.size() / 2];
        std::cout << "  refined:   median " << med << " Hz ("
                  << std::showpos << 1200.0f * std::log2(med / nominal)
                  << std::noshowpos << " cents from nominal), range "
                  << sorted.front() << " - " << sorted.back() << " Hz\n";
    }
    std::cout << "  controls:  " << T << " frames x (" << K << " harmonics + "
              << NOISE_BANDS << " noise bands) = " << T * (K + NOISE_BANDS)
              << " values\n";

    // Raw (pre-activation) controls: these are what Adam updates.
    //
    // They are NOT stored once per control frame.  They are stored at
    // "knots" spaced S frames apart and interpolated linearly in between,
    // with S chosen so that a knot spans about four periods of the note.
    // The reason is a degeneracy: if a harmonic's amplitude may change as
    // fast as the waveform itself repeats, switching it on and off becomes
    // another way of shaping the waveform, and the optimiser uses it --
    // the fit "flickers" instead of settling on steady amplitudes.  With
    // a 512-sample hop (11.6 ms) that happens for low notes (A2 has a
    // 9 ms period); for A4 the rule gives S = 1 and nothing changes.
    const std::size_t S = (argc > 9) ? static_cast<std::size_t>(std::atoi(argv[9]))
        : std::max<std::size_t>(1, static_cast<std::size_t>(
              std::ceil(4.0f * sr / (nominal * float(HOP)))));
    const std::size_t Tk = (T - 1) / S + 2;          // knots
    Matrix<float> knotA(K, Tk, -4.0f), knotM(NOISE_BANDS, Tk, -4.0f);
    Matrix<float> rawA(K, T), rawM(NOISE_BANDS, T);
    auto expand = [&](const Matrix<float>& knots, Matrix<float>& raw) {
        for (std::size_t t = 0; t < T; ++t) {
            const std::size_t j = t / S;
            const float fr = float(t % S) / float(S);
            for (std::size_t r = 0; r < raw.rows(); ++r)
                raw(r, t) = (1.0f - fr) * knots(r, j) + fr * knots(r, j + 1);
        }
    };
    auto collapse = [&](const Matrix<float>& draw) {     // adjoint of expand
        Matrix<float> dk(draw.rows(), Tk, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            const std::size_t j = t / S;
            const float fr = float(t % S) / float(S);
            for (std::size_t r = 0; r < draw.rows(); ++r) {
                dk(r, j)     += (1.0f - fr) * draw(r, t);
                dk(r, j + 1) += fr * draw(r, t);
            }
        }
        return dk;
    };
    std::cout << "  knots:     every " << S << " frame(s) = "
              << std::setprecision(1) << 1000.0f * S * HOP / sr << " ms ("
              << std::setprecision(2) << S * HOP * nominal / sr
              << " periods of the note)\n\n";
    c.A = Matrix<float>(K, T);
    c.M = Matrix<float>(NOISE_BANDS, T);

    HarmonicSynth<float> synth(sr, HOP);
    FilteredNoise<float> noise(HOP, NOISE_BANDS);
    // FFT sizes for the loss.  A window shorter than a few periods of the
    // note cannot separate neighbouring harmonics; what it measures then
    // depends on their relative PHASES, which this synthesiser does not
    // control, and the fit trades correct amplitudes for a better waveform
    // shape.  So we use the five smallest power-of-two sizes that still
    // span at least 2.5 periods (capped at 8192): 256..4096 for A4,
    // 1024..8192 for A2.
    if (fft_sizes.empty()) {
        const float period = sr / nominal;
        for (std::size_t N = 64; N <= 8192 && fft_sizes.size() < 5; N *= 2)
            if (float(N) >= 2.5f * period) fft_sizes.push_back(N);
        if (fft_sizes.empty()) fft_sizes.push_back(8192);
    }
    std::cout << "  loss FFTs: ";
    for (std::size_t N : fft_sizes) std::cout << N << " ";
    std::cout << "\n";
    SpectralLoss<float>  loss(fft_sizes);
    loss.set_target(target);
    const float lr = (argc > 7) ? std::stof(argv[7]) : 0.1f;
    const float adam_eps = (argc > 8) ? std::stof(argv[8]) : 1e-8f;
    Adam<float> opt(lr, 0.9f, 0.999f, adam_eps);

    std::cout << "iter    loss     time\n----    ------   -----\n";
    const auto t0 = clock::now();
    float last = 0.0f;
    for (int it = 0; it <= iterations; ++it) {
        expand(knotA, rawA);
        expand(knotM, rawM);
        c.A = rawA.apply(act);
        c.M = rawM.apply(act);
        auto y = synth.forward(c.f0, c.A, n);
        const auto z = noise.forward(c.M, n);
        for (std::size_t i = 0; i < n; ++i) y[i] += z[i];
        last = loss.forward(y);

        if (it % 25 == 0 || it == iterations) {
            const double secs =
                std::chrono::duration<double>(clock::now() - t0).count();
            std::cout << std::setw(4) << it << "    " << std::setprecision(4)
                      << last << "   " << std::setprecision(1) << secs << "s\n"
                      << std::flush;
        }
        if (it == iterations) break;

        const auto dy = loss.backward();
        Matrix<float> dA = synth.backward(dy);
        Matrix<float> dM = noise.backward(dy);
        // Chain through the output activation.
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t k = 0; k < K; ++k) dA(k, t) *= act_deriv(rawA(k, t));
            for (std::size_t b = 0; b < NOISE_BANDS; ++b)
                dM(b, t) *= act_deriv(rawM(b, t));
        }
        Matrix<float> dkA = collapse(dA), dkM = collapse(dM);
        opt.step(knotA, dkA);
        opt.step(knotM, dkM);
    }

    std::vector<float> harmonic, noise_part;
    const auto y = render(c, &harmonic, &noise_part);
    auto energy = [](const std::vector<float>& v) {
        double e = 0.0;
        for (float s : v) e += double(s) * s;
        return e;
    };
    std::cout << "\nfinal loss " << std::setprecision(4) << last
              << ";  energy: target " << energy(target) << ", resynthesis "
              << energy(y) << " (harmonic " << energy(harmonic) << ", noise "
              << energy(noise_part) << ")\n";

    save_wav(prefix + "_resynth.wav",  y,          wav.sample_rate);
    save_wav(prefix + "_harmonic.wav", harmonic,   wav.sample_rate);
    save_wav(prefix + "_noise.wav",    noise_part, wav.sample_rate);
    save_controls(prefix + ".ctl", c);
    std::cout << "wrote " << prefix << "_resynth.wav, _harmonic.wav, _noise.wav, .ctl\n";
    return 0;
}

int cmd_mix(int argc, char** argv, bool glide) {
    if (argc < 5) {
        std::cerr << "usage: synth_fit " << (glide ? "morph" : "blend")
                  << " <a.ctl> <b.ctl> <out.wav>" << (glide ? "" : " [mix]") << "\n";
        return 1;
    }
    const Controls a = load_controls(argv[2]);
    const Controls b = load_controls(argv[3]);
    const float fixed = (!glide && argc > 5) ? std::stof(argv[5]) : 0.5f;
    const Controls c = combine(a, b, [&](std::size_t t, std::size_t T) {
        if (!glide) return fixed;
        // Hold A for the first fifth, B for the last fifth, glide between.
        const float x = (float(t) / float(T - 1) - 0.2f) / 0.6f;
        return std::min(1.0f, std::max(0.0f, x));
    });
    save_wav(argv[4], render(c), static_cast<std::uint32_t>(c.sample_rate));
    std::cout << "wrote " << argv[4] << "  (" << c.n_samples / c.sample_rate << " s)\n";
    return 0;
}

int cmd_render(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: synth_fit render <in.ctl> <out.wav> "
                     "[semitones] [glide_semitones]\n";
        return 1;
    }
    Controls c = load_controls(argv[2]);
    const float shift = (argc > 4) ? std::stof(argv[4]) : 0.0f;
    const float glide = (argc > 5) ? std::stof(argv[5]) : 0.0f;
    const std::size_t T = c.f0.size();
    for (std::size_t t = 0; t < T; ++t) {
        const float st = shift + glide * float(t) / float(T - 1);
        c.f0[t] *= std::pow(2.0f, st / 12.0f);
    }
    save_wav(argv[3], render(c), static_cast<std::uint32_t>(c.sample_rate));
    std::cout << "wrote " << argv[3] << "  (transposed " << shift
              << " semitones, glide " << glide << ")\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = (argc > 1) ? argv[1] : "";
    try {
        if (cmd == "fit")    return cmd_fit(argc, argv);
        if (cmd == "morph")  return cmd_mix(argc, argv, true);
        if (cmd == "blend")  return cmd_mix(argc, argv, false);
        if (cmd == "render") return cmd_render(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage:\n"
                 "  synth_fit fit    <in.wav> <note|Hz> <prefix> [iterations]\n"
                 "  synth_fit morph  <a.ctl> <b.ctl> <out.wav>\n"
                 "  synth_fit blend  <a.ctl> <b.ctl> <out.wav> [mix]\n"
                 "  synth_fit render <in.ctl> <out.wav> [semitones] [glide_semitones]\n";
    return 1;
}
