// vae_generate / vae_generate.cpp
//
// Generative applications of the audio VAE.  Two modes:
//
//   interp  -- morph between two sounds by interpolating their latent
//              codes and decoding.  Reuses sound A's phase frame by frame
//              (donor = A), so no phase reconstruction is needed.  With no
//              t given, the morph sweeps A -> B over the duration; with a
//              fixed t in [0,1] it holds a static blend.
//
//   sample  -- draw z ~ N(0, I), decode to a single invented magnitude
//              spectrum, and sustain it using the PHASE of a donor sound.
//              This is the crudest path: the donor's phase only "fits" the
//              bins where the donor had energy, so expect artifacts.  It is
//              the case Griffin-Lim (a later refinement) would most help.
//
// Why no Griffin-Lim: the VAE models magnitude only, so any generated
// magnitude needs a phase from somewhere.  Interpolation has a real source
// (A) to borrow from; sampling has none, so we borrow a donor's.  Both
// avoid iterative phase estimation -- at a cost in fidelity for sampling.
//
// The architecture rebuilt here MUST match train_audio_vae.
//
// Usage:
//   vae_generate <model_prefix> interp <A.wav> <B.wav> <out.wav> [t]
//   vae_generate <model_prefix> sample <donor.wav> <out.wav> [seed]
//
#include "Dense.h"
#include "Network.h"
#include "ReLU.h"
#include "STFT.h"
#include "Wav.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

constexpr std::size_t FRAME_SIZE = 4096;
constexpr std::size_t HOP_SIZE   = 2048;
constexpr std::size_t FEAT_DIM   = FRAME_SIZE / 2 + 1;   // 2049
constexpr std::size_t LATENT     = 32;

// Rebuild the train_audio_vae architecture and load weights.
static void load_models(const std::string& prefix,
                        Network<float>& enc, Network<float>& dec) {
    enc.add<Dense>(FEAT_DIM, 256);
    enc.add<ReLU>();
    enc.add<Dense>(256, 64);
    enc.add<ReLU>();
    enc.add<Dense>(64, 2 * LATENT);

    dec.add<Dense>(LATENT, 64);
    dec.add<ReLU>();
    dec.add<Dense>(64, 256);
    dec.add<ReLU>();
    dec.add<Dense>(256, FEAT_DIM);

    enc.load(prefix + ".enc");
    dec.load(prefix + ".dec");
    enc.eval();
    dec.eval();
}

// STFT a signal -> (log-magnitude matrix 2049 x n_frames, per-frame phases).
static Matrix<float> stft_logmag(const std::vector<float>& samples,
                                 std::vector<std::vector<float>>& phases_out) {
    auto frames = stft(samples, FRAME_SIZE, HOP_SIZE);
    const std::size_t n = frames.size();
    Matrix<float> lm(FEAT_DIM, n == 0 ? 1 : n);
    phases_out.resize(n);
    for (std::size_t f = 0; f < n; ++f) {
        auto mag = magnitude(frames[f]);
        phases_out[f] = phase(frames[f]);
        for (std::size_t i = 0; i < FEAT_DIM; ++i)
            lm(i, f) = std::log(1.0f + mag[i]);
    }
    return lm;
}

// Encode to the latent MEAN (top LATENT rows of the encoder output).
static Matrix<float> encode_mu(Network<float>& enc, const Matrix<float>& logmag) {
    Matrix<float> h = enc.forward(logmag);
    const std::size_t n = logmag.cols();
    Matrix<float> mu(LATENT, n);
    for (std::size_t d = 0; d < LATENT; ++d)
        for (std::size_t f = 0; f < n; ++f)
            mu(d, f) = h(d, f);
    return mu;
}

// Decode latent codes to magnitudes (invert log: clamp >= 0, then expm1).
static Matrix<float> decode_mag(Network<float>& dec, const Matrix<float>& z) {
    Matrix<float> r = dec.forward(z);
    for (std::size_t i = 0; i < r.rows(); ++i)
        for (std::size_t j = 0; j < r.cols(); ++j) {
            const float m = std::expm1(std::max(0.0f, r(i, j)));
            r(i, j) = m < 0.0f ? 0.0f : m;
        }
    return r;
}

static void write_normalized(const std::string& path,
                             std::vector<float>& out, std::uint32_t sr) {
    float peak = 0.0f;
    for (float s : out) peak = std::max(peak, std::fabs(s));
    if (peak > 1e-6f) {
        const float g = 0.9f / peak;
        for (float& s : out) s *= g;
    }
    save_wav(path, out, sr);
}

static int do_interp(Network<float>& enc, Network<float>& dec,
                     const std::string& a_path, const std::string& b_path,
                     const std::string& out_path, float fixed_t) {
    WavData A = load_wav(a_path), B = load_wav(b_path);
    std::cout << "A: " << a_path << "  B: " << b_path << "\n";

    std::vector<std::vector<float>> phA, phB;
    Matrix<float> lmA = stft_logmag(A.samples, phA);
    Matrix<float> lmB = stft_logmag(B.samples, phB);
    const std::size_t nA = phA.size(), nB = phB.size();
    if (nA == 0 || nB == 0) { std::cerr << "a sound is too short\n"; return 1; }

    Matrix<float> muA = encode_mu(enc, lmA);
    Matrix<float> muB = encode_mu(enc, lmB);

    if (fixed_t >= 0.0f)
        std::cout << "static blend at t=" << fixed_t << " (donor phase = A)\n";
    else
        std::cout << "morph sweep A -> B over " << nA << " frames (donor phase = A)\n";

    // Build the interpolated latent trajectory on A's timeline.
    Matrix<float> z(LATENT, nA);
    for (std::size_t f = 0; f < nA; ++f) {
        const float t = (fixed_t >= 0.0f)
                      ? fixed_t
                      : (nA > 1 ? static_cast<float>(f) / (nA - 1) : 0.0f);
        // Map A-frame f onto B's timeline.
        std::size_t fB = (nA > 1)
            ? static_cast<std::size_t>(std::lround(
                  static_cast<double>(f) * (nB - 1) / (nA - 1)))
            : 0;
        if (fB >= nB) fB = nB - 1;
        for (std::size_t d = 0; d < LATENT; ++d)
            z(d, f) = (1.0f - t) * muA(d, f) + t * muB(d, fB);
    }

    Matrix<float> mag = decode_mag(dec, z);

    std::vector<std::vector<std::complex<float>>> frames(nA);
    for (std::size_t f = 0; f < nA; ++f) {
        std::vector<float> col(FEAT_DIM);
        for (std::size_t i = 0; i < FEAT_DIM; ++i) col[i] = mag(i, f);
        frames[f] = rebuild_hermitian(col, phA[f]);    // A's phase
    }
    auto out = istft(frames, FRAME_SIZE, HOP_SIZE, A.samples.size());
    write_normalized(out_path, out, A.sample_rate);
    std::cout << "wrote " << out_path << "\n";
    return 0;
}

static int do_sample(Network<float>& dec,
                     const std::string& donor_path, const std::string& out_path,
                     unsigned seed) {
    WavData donor = load_wav(donor_path);
    std::vector<std::vector<float>> phD;
    stft_logmag(donor.samples, phD);          // we only need the donor's phase
    const std::size_t nD = phD.size();
    if (nD == 0) { std::cerr << "donor too short\n"; return 1; }

    std::cout << "sampling z ~ N(0, I)  (seed=" << seed
              << "), donor phase from " << donor_path << "\n";

    Matrix<float> z(LATENT, 1);
    z.randomizeNormal(0.0f, 1.0f, seed);
    Matrix<float> mag = decode_mag(dec, z);   // (FEAT_DIM, 1): one invented timbre

    std::vector<float> col(FEAT_DIM);
    for (std::size_t i = 0; i < FEAT_DIM; ++i) col[i] = mag(i, 0);

    // Sustain the single timbre, advancing the donor's phase frame by frame.
    std::vector<std::vector<std::complex<float>>> frames(nD);
    for (std::size_t f = 0; f < nD; ++f) frames[f] = rebuild_hermitian(col, phD[f]);

    auto out = istft(frames, FRAME_SIZE, HOP_SIZE, donor.samples.size());
    write_normalized(out_path, out, donor.sample_rate);
    std::cout << "wrote " << out_path
              << "  (sampling uses donor phase; Griffin-Lim would sharpen it)\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n"
                  << "  vae_generate <prefix> interp <A.wav> <B.wav> <out.wav> [t]\n"
                  << "  vae_generate <prefix> sample <donor.wav> <out.wav> [seed]\n";
        return 1;
    }
    const std::string prefix = argv[1];
    const std::string mode   = argv[2];

    std::cout << "weft :: vae_generate\n";
    Network<float> enc, dec;
    try { load_models(prefix, enc, dec); }
    catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "loaded model: " << prefix << ".enc / .dec\n";

    try {
        if (mode == "interp") {
            if (argc < 6) { std::cerr << "interp needs <A> <B> <out> [t]\n"; return 1; }
            const float t = (argc > 6) ? std::atof(argv[6]) : -1.0f;
            return do_interp(enc, dec, argv[3], argv[4], argv[5], t);
        } else if (mode == "sample") {
            if (argc < 5) { std::cerr << "sample needs <donor> <out> [seed]\n"; return 1; }
            const unsigned seed = (argc > 5)
                ? static_cast<unsigned>(std::atoi(argv[5])) : 1u;
            return do_sample(dec, argv[3], argv[4], seed);
        }
        std::cerr << "unknown mode: " << mode << " (use interp or sample)\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
