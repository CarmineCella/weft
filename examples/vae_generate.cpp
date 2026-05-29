// vae_generate / vae_generate.cpp
//
// Generative applications of the audio VAE.  Two modes:
//
//   interp  -- morph between two sounds by interpolating their latent
//              codes and decoding.  Reuses sound A's phase frame by frame
//              (donor = A), so no phase reconstruction is needed.  Mix
//              parameter t sweeps continuously from 0 (full A) at the
//              first output frame to 1 (full B) at the last, decoding the
//              latent z = (1-t)*z_A + t*z_B at every frame in between.
//
//   sample  -- draw N anchor latents z ~ N(0, I) and walk linearly
//              through them across the duration of a donor sound,
//              decoding to a smooth trajectory of invented magnitudes.
//              The donor supplies phase frame by frame.  Hearing several
//              samples in sequence also shows the latent space's
//              continuity -- nearby z's decode to similar timbres.
//              N defaults to 4.
//
// Why no Griffin-Lim: the VAE models magnitude only, so any generated
// magnitude needs a phase from somewhere.  Interpolation has a real
// source (A) to borrow from; sampling borrows a donor's.  Both avoid
// iterative phase estimation -- at a cost in fidelity for sampling.
//
// A short cosine fade-in/out is applied at the output to suppress the
// click that the windowed iSTFT can produce at the very edges, where
// the synthesis-window power is near zero.
//
// The architecture rebuilt here MUST match train_audio_vae.
//
// Usage:
//   vae_generate <model_prefix> interp <A.wav> <B.wav> <out.wav>
//   vae_generate <model_prefix> sample <donor.wav> <out.wav> [count] [seed]
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
    // (1) Cosine fade-in/out (~46 ms at 44.1 kHz) to suppress the click
    // the iSTFT can produce at the edges: where the synthesis-window power
    // is small but above the divide-by-zero threshold, the WOLA
    // normalisation amplifies tiny errors into an audible transient.  One
    // hop's worth of fade comfortably covers that region.
    {
        const std::size_t fade = std::min<std::size_t>(2048, out.size() / 4);
        for (std::size_t i = 0; i < fade; ++i) {
            const float w = 0.5f * (1.0f - std::cos(3.14159265f * i / fade));
            out[i]                  *= w;
            out[out.size() - 1 - i] *= w;
        }
    }
    // (2) Percentile-based normalisation.  A plain peak-to-0.9 normaliser
    // is hostage to any single transient (residual click, model artifact),
    // which can pull the rest of the signal down to 1/100 of its actual
    // level.  Using the 99th percentile of |x| as the "peak" ignores those
    // outliers; samples that exceed 1.0 after the gain are clipped on
    // write.  This is what audio normalisation tools do.
    if (!out.empty()) {
        std::vector<float> abs_out(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) abs_out[i] = std::fabs(out[i]);
        const std::size_t p =
            static_cast<std::size_t>(0.99 * (abs_out.size() - 1));
        std::nth_element(abs_out.begin(), abs_out.begin() + p, abs_out.end());
        const float peak = abs_out[p];
        if (peak > 1e-6f) {
            const float g = 0.9f / peak;
            for (float& s : out) s *= g;
        }
    }
    save_wav(path, out, sr);
}

static int do_interp(Network<float>& enc, Network<float>& dec,
                     const std::string& a_path, const std::string& b_path,
                     const std::string& out_path) {
    WavData A = load_wav(a_path), B = load_wav(b_path);
    std::cout << "A: " << a_path << "  B: " << b_path << "\n";

    std::vector<std::vector<float>> phA, phB;
    Matrix<float> lmA = stft_logmag(A.samples, phA);
    Matrix<float> lmB = stft_logmag(B.samples, phB);
    const std::size_t nA = phA.size(), nB = phB.size();
    if (nA == 0 || nB == 0) { std::cerr << "a sound is too short\n"; return 1; }

    Matrix<float> muA = encode_mu(enc, lmA);
    Matrix<float> muB = encode_mu(enc, lmB);

    std::cout << "continuous morph A -> B over " << nA
              << " frames (t sweeps linearly 0->1, donor phase = A)\n";

    // Linear sweep: t = 0 at the first frame, 1 at the last, no holds, no
    // fixed blend.  z(f) = (1-t)*muA(f) + t*muB(f') with f' = the same
    // fractional position in B's timeline (time-stretches B to A's length).
    Matrix<float> z(LATENT, nA);
    for (std::size_t f = 0; f < nA; ++f) {
        const float t = (nA > 1)
            ? static_cast<float>(f) / static_cast<float>(nA - 1)
            : 0.0f;
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
                     std::size_t count, unsigned seed) {
    if (count < 1) count = 1;

    WavData donor = load_wav(donor_path);
    std::vector<std::vector<float>> phD;
    stft_logmag(donor.samples, phD);          // we only need the donor's phase
    const std::size_t nD = phD.size();
    if (nD == 0) { std::cerr << "donor too short\n"; return 1; }

    std::cout << "sampling " << count << " anchor latent(s) ~ N(0, I)"
              << "  (seed=" << seed << "), donor phase from " << donor_path
              << "\n";

    // Sample `count` anchor latents.  randomizeNormal fills the whole
    // matrix in one shot, so each column gets independent draws.
    Matrix<float> anchors(LATENT, count);
    anchors.randomizeNormal(0.0f, 1.0f, seed);

    // Build a smooth trajectory of length nD that linearly walks through
    // the anchors -- frame 0 lands on anchor 0, the last frame on anchor
    // (count-1), and interior frames interpolate between adjacent anchors.
    // One sample (count == 1) degenerates to sustaining that single timbre.
    Matrix<float> z(LATENT, nD);
    for (std::size_t f = 0; f < nD; ++f) {
        float u;
        if (count == 1 || nD == 1) {
            u = 0.0f;
        } else {
            u = static_cast<float>(f) * (count - 1) / (nD - 1);    // 0 .. count-1
        }
        const std::size_t k    = std::min(static_cast<std::size_t>(u), count - 1);
        const std::size_t k1   = std::min(k + 1, count - 1);
        const float       frac = u - static_cast<float>(k);
        for (std::size_t d = 0; d < LATENT; ++d)
            z(d, f) = (1.0f - frac) * anchors(d, k) + frac * anchors(d, k1);
    }

    Matrix<float> mag = decode_mag(dec, z);   // (FEAT_DIM, nD)

    std::vector<std::vector<std::complex<float>>> frames(nD);
    for (std::size_t f = 0; f < nD; ++f) {
        std::vector<float> col(FEAT_DIM);
        for (std::size_t i = 0; i < FEAT_DIM; ++i) col[i] = mag(i, f);
        frames[f] = rebuild_hermitian(col, phD[f]);
    }
    auto out = istft(frames, FRAME_SIZE, HOP_SIZE, donor.samples.size());
    write_normalized(out_path, out, donor.sample_rate);
    std::cout << "wrote " << out_path
              << "  (sampling uses donor phase; Griffin-Lim would sharpen it)\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n"
                  << "  vae_generate <prefix> interp <A.wav> <B.wav> <out.wav>\n"
                  << "  vae_generate <prefix> sample <donor.wav> <out.wav> "
                     "[count] [seed]\n";
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
            if (argc < 6) { std::cerr << "interp needs <A> <B> <out>\n"; return 1; }
            return do_interp(enc, dec, argv[3], argv[4], argv[5]);
        } else if (mode == "sample") {
            if (argc < 5) {
                std::cerr << "sample needs <donor> <out> [count] [seed]\n";
                return 1;
            }
            const std::size_t count = (argc > 5)
                ? std::max(1, std::atoi(argv[5])) : 4;
            const unsigned    seed  = (argc > 6)
                ? static_cast<unsigned>(std::atoi(argv[6])) : 1u;
            return do_sample(dec, argv[3], argv[4], count, seed);
        }
        std::cerr << "unknown mode: " << mode << " (use interp or sample)\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
