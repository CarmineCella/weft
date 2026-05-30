// convae_generate.cpp
//
// Generative applications of the convolutional audio VAE.  Same spirit
// as vae_generate, but the ConvVAE works on (1, FREQ_BINS, TIME_FRAMES)
// patches of the log-magnitude spectrogram rather than single frames,
// so the latent code encodes a short *time slice* of timbre, not an
// instantaneous spectrum.
//
// Two modes:
//
//   interp  -- morph between two sounds by latent interpolation.  Each
//              input wav's MIDDLE patch is encoded to a single latent
//              z; we then synthesise N patches at evenly spaced t
//              values, concatenating their frames into a longer
//              spectrogram.  Phase is borrowed from sound A throughout
//              (tiled if A is shorter than the output).  The audible
//              result: A's rhythmic character with timbre that morphs
//              continuously toward B over the output.
//
//   sample  -- draw count anchor latents z ~ N(0, I) and walk through
//              them over N patches.  Donor wav supplies the phase
//              (also tiled).  Sampling one anchor (count=1) sustains
//              a single invented timbre; sampling several walks a
//              trajectory through the learned timbre manifold.
//
// Why no Griffin-Lim: the VAE models magnitude only, so any decoded
// magnitude needs a phase from somewhere.  Borrowing a donor's phase
// is fast and produces musically coherent output; Griffin-Lim would
// sharpen the result at the cost of an iterative optimisation we
// don't have in the library yet.
//
// The architecture rebuilt here MUST match train_conv_audio_vae.
//
// Usage:
//   convae_generate <prefix> interp <A.wav> <B.wav> <out.wav> [N_patches]
//   convae_generate <prefix> sample <donor.wav> <out.wav> [count] [N_patches] [seed]
//
//   prefix     loads <prefix>.{enc_conv,enc_dense,dec_dense,dec_conv}
//   N_patches  number of latent-space waypoints (default 8 = ~6 sec)
//   count      for sample mode: number of anchor latents (default 4)

#include "Conv2D.h"
#include "ConvNetwork.h"
#include "Dense.h"
#include "Flatten.h"
#include "Matrix.h"
#include "MaxPool2D.h"
#include "Network.h"
#include "ReLU.h"
#include "ReLU4D.h"
#include "STFT.h"
#include "MelTransform.h"
#include "Tensor4D.h"
#include "Upsample2D.h"
#include "Wav.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

// ---- spectrogram + patch constants (must match train_conv_audio_vae) -----
constexpr std::size_t FRAME_SIZE   = 4096;
constexpr std::size_t HOP_SIZE     = 2048;
constexpr std::size_t FREQ_BINS_IN = 2049;
constexpr std::size_t FREQ_BINS    = 64;
constexpr std::size_t TIME_FRAMES  = 16;
constexpr std::size_t LATENT       = 32;
constexpr std::size_t FLAT         = 64 * 8 * 2;   // bottleneck shape (64, 8, 2)

// --------------------------------------------------------------------------
// Rebuild the train_conv_audio_vae architecture and load weights from the
// four model files.  Switching all four networks to eval() turns off
// dropout etc. (we don't use any in the ConvVAE, but it's hygiene).
// --------------------------------------------------------------------------
static void load_models(const std::string& prefix,
                        ConvNetwork<float>& enc_conv,
                        Network<float>&     enc_dense,
                        Network<float>&     dec_dense,
                        ConvNetwork<float>& dec_conv) {
    enc_conv.add<Conv2D>  (1,  16, 3, 1, 1);
    enc_conv.add<ReLU4D>  ();
    enc_conv.add<MaxPool2D>(2);
    enc_conv.add<Conv2D>  (16, 32, 3, 1, 1);
    enc_conv.add<ReLU4D>  ();
    enc_conv.add<MaxPool2D>(2);
    enc_conv.add<Conv2D>  (32, 64, 3, 1, 1);
    enc_conv.add<ReLU4D>  ();
    enc_conv.add<MaxPool2D>(2);

    enc_dense.add<Dense>(FLAT, 256);
    enc_dense.add<ReLU>();
    enc_dense.add<Dense>(256, 2 * LATENT);

    dec_dense.add<Dense>(LATENT, 256);
    dec_dense.add<ReLU>();
    dec_dense.add<Dense>(256, FLAT);
    dec_dense.add<ReLU>();

    dec_conv.add<Upsample2D>(2);
    dec_conv.add<Conv2D>    (64, 32, 3, 1, 1);
    dec_conv.add<ReLU4D>    ();
    dec_conv.add<Upsample2D>(2);
    dec_conv.add<Conv2D>    (32, 16, 3, 1, 1);
    dec_conv.add<ReLU4D>    ();
    dec_conv.add<Upsample2D>(2);
    dec_conv.add<Conv2D>    (16,  1, 3, 1, 1);

    enc_conv .load(prefix + ".enc_conv");
    enc_dense.load(prefix + ".enc_dense");
    dec_dense.load(prefix + ".dec_dense");
    dec_conv .load(prefix + ".dec_conv");
    enc_conv.eval(); enc_dense.eval();
    dec_dense.eval(); dec_conv.eval();
}

// --------------------------------------------------------------------------
// STFT a signal, split into per-frame log-magnitudes and phases.  The full
// 2049-bin log-magnitude is kept here; we'll downsample to 64 bins per
// encoded patch but keep the full-res phase for inversion.
// --------------------------------------------------------------------------
static void stft_decomp(const std::vector<float>& samples,
                        std::vector<std::vector<float>>& logmag_out,
                        std::vector<std::vector<float>>& phases_out) {
    auto frames = stft(samples, FRAME_SIZE, HOP_SIZE);
    const std::size_t n = frames.size();
    logmag_out.assign(n, std::vector<float>(FREQ_BINS_IN));
    phases_out.assign(n, std::vector<float>(FREQ_BINS_IN));
    for (std::size_t f = 0; f < n; ++f) {
        auto mag = magnitude(frames[f]);
        phases_out[f] = phase(frames[f]);
        for (std::size_t i = 0; i < FREQ_BINS_IN; ++i)
            logmag_out[f][i] = std::log1p(mag[i]);
    }
}

// Take the MIDDLE TIME_FRAMES-frame patch from a full per-frame log-mag.
// Pads with zeros at the edges if the input is shorter than TIME_FRAMES.
// Returns a (1, 1, FREQ_BINS, TIME_FRAMES) tensor (mel-transformed in freq).
static Tensor4D<float>
middle_patch(const std::vector<std::vector<float>>& logmag,
             const MelTransform<float>& mel) {
    const std::size_t n = logmag.size();
    Tensor4D<float> P(1, 1, FREQ_BINS, TIME_FRAMES);
    // Centre the patch on the middle frame.
    const std::size_t centre = n / 2;
    const std::size_t half   = TIME_FRAMES / 2;
    const long start = static_cast<long>(centre) - static_cast<long>(half);

    std::vector<float> col_lo(FREQ_BINS);
    for (std::size_t t = 0; t < TIME_FRAMES; ++t) {
        const long src = start + static_cast<long>(t);
        if (src < 0 || src >= static_cast<long>(n)) {
            for (std::size_t f = 0; f < FREQ_BINS; ++f)
                P(0, 0, f, t) = 0.0f;
            continue;
        }
        mel.linear_logmag_to_logmel(logmag[src].data(), col_lo.data());
        for (std::size_t f = 0; f < FREQ_BINS; ++f)
            P(0, 0, f, t) = col_lo[f];
    }
    return P;
}

// Encode a single patch to its latent MEAN (mu only, no sampling).
static std::vector<float> encode_mu(ConvNetwork<float>& enc_conv,
                                    Network<float>&     enc_dense,
                                    const Tensor4D<float>& patch) {
    Tensor4D<float> feat = enc_conv.forward(patch);
    Matrix<float>   flat = flatten(feat);
    Matrix<float>   h    = enc_dense.forward(flat);
    std::vector<float> mu(LATENT);
    for (std::size_t d = 0; d < LATENT; ++d) mu[d] = h(d, 0);
    return mu;
}

// Decode a batch of latents (LATENT x B) to a batch of patches
// (B, 1, FREQ_BINS, TIME_FRAMES).
static Tensor4D<float> decode_z(Network<float>&     dec_dense,
                                ConvNetwork<float>& dec_conv,
                                const Matrix<float>& z) {
    const std::size_t B = z.cols();
    Matrix<float>   dec_flat = dec_dense.forward(z);
    Tensor4D<float> dec_feat = unflatten(dec_flat, B, 64, 8, 2);
    return dec_conv.forward(dec_feat);
}

// Concatenate N decoded patches into a single sequence of TIME_FRAMES*N
// low-res log-magnitude columns.  patches has shape (N, 1, FREQ, TIME).
static std::vector<std::vector<float>>
patches_to_columns(const Tensor4D<float>& patches) {
    const std::size_t N = patches.N();
    std::vector<std::vector<float>> cols(N * TIME_FRAMES, std::vector<float>(FREQ_BINS));
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t t = 0; t < TIME_FRAMES; ++t)
            for (std::size_t f = 0; f < FREQ_BINS; ++f)
                cols[n * TIME_FRAMES + t][f] = patches(n, 0, f, t);
    return cols;
}

// Convert each 64-bin log-mel column to a 2049-bin LINEAR magnitude column
// (via the mel transform's pseudo-inverse), pair with a donor phase
// column, return complex frames ready for iSTFT.
//
// donor_phase is tiled across the output if it's shorter.  That tiling
// gives the output audio its rhythmic character while the magnitude
// defines its timbre.
static std::vector<std::vector<std::complex<float>>>
build_frames(const std::vector<std::vector<float>>& logmel_lo,
             const std::vector<std::vector<float>>& donor_phase,
             const MelTransform<float>& mel) {
    const std::size_t n_out = logmel_lo.size();
    std::vector<std::vector<std::complex<float>>> frames(n_out);
    std::vector<float> col(FREQ_BINS_IN);
    for (std::size_t f = 0; f < n_out; ++f) {
        mel.logmel_to_linear_mag(logmel_lo[f].data(), col.data());
        const std::size_t ph_idx = f % donor_phase.size();
        frames[f] = rebuild_hermitian(col, donor_phase[ph_idx]);
    }
    return frames;
}

// Per-vae_generate convention: short cosine fade-in/out and percentile-
// based normalisation to suppress edge clicks and avoid one transient
// pulling the whole signal down to 1/100 of its actual level.
static void write_normalized(const std::string& path,
                             std::vector<float>& out, std::uint32_t sr) {
    {
        const std::size_t fade = std::min<std::size_t>(2048, out.size() / 4);
        for (std::size_t i = 0; i < fade; ++i) {
            const float w = 0.5f * (1.0f - std::cos(3.14159265f * i / fade));
            out[i]                  *= w;
            out[out.size() - 1 - i] *= w;
        }
    }
    if (!out.empty()) {
        std::vector<float> abs_out(out.size());
        for (std::size_t i = 0; i < out.size(); ++i)
            abs_out[i] = std::fabs(out[i]);
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

// --------------------------------------------------------------------------
// interp: encode A's middle patch and B's middle patch, sweep latent
// linearly between them across N output patches, decode, concatenate
// frames, borrow A's phase (tiled), iSTFT, write WAV.
// --------------------------------------------------------------------------
static int do_interp(ConvNetwork<float>& enc_conv,
                     Network<float>&     enc_dense,
                     Network<float>&     dec_dense,
                     ConvNetwork<float>& dec_conv,
                     const MelTransform<float>& mel,
                     const std::string& a_path,
                     const std::string& b_path,
                     const std::string& out_path,
                     std::size_t N_patches) {
    WavData A = load_wav(a_path), B = load_wav(b_path);
    std::cout << "A: " << a_path << "\nB: " << b_path << "\n";

    std::vector<std::vector<float>> lmA, lmB, phA, phB;
    stft_decomp(A.samples, lmA, phA);
    stft_decomp(B.samples, lmB, phB);
    if (phA.empty() || phB.empty()) {
        std::cerr << "input sound too short for STFT\n"; return 1;
    }
    if (lmA.size() < TIME_FRAMES || lmB.size() < TIME_FRAMES) {
        std::cerr << "warning: input shorter than " << TIME_FRAMES
                  << " frames; encoded patch is zero-padded\n";
    }

    auto z_A = encode_mu(enc_conv, enc_dense, middle_patch(lmA, mel));
    auto z_B = encode_mu(enc_conv, enc_dense, middle_patch(lmB, mel));

    std::cout << "morphing A -> B over " << N_patches
              << " patches (~" << std::fixed
              << static_cast<double>(N_patches) * TIME_FRAMES *
                 HOP_SIZE / A.sample_rate
              << "s output, donor phase = A)\n";

    // Build z matrix (LATENT x N_patches), each column an interpolation
    // step between z_A (t=0) and z_B (t=1).
    Matrix<float> Z(LATENT, N_patches);
    for (std::size_t k = 0; k < N_patches; ++k) {
        const float t = (N_patches > 1)
            ? static_cast<float>(k) / static_cast<float>(N_patches - 1)
            : 0.0f;
        for (std::size_t d = 0; d < LATENT; ++d)
            Z(d, k) = (1.0f - t) * z_A[d] + t * z_B[d];
    }

    Tensor4D<float> patches = decode_z(dec_dense, dec_conv, Z);
    auto cols  = patches_to_columns(patches);              // N_patches*TIME columns
    auto frames = build_frames(cols, phA, mel);                 // tile A's phase
    auto out   = istft(frames, FRAME_SIZE, HOP_SIZE, 0);
    write_normalized(out_path, out, A.sample_rate);
    std::cout << "wrote " << out_path << "\n";
    return 0;
}

// --------------------------------------------------------------------------
// sample: draw `count` anchor latents from N(0, I), walk linearly through
// them across N_patches output patches, decode, build frames with the
// donor's phase tiled, iSTFT, write WAV.
// --------------------------------------------------------------------------
static int do_sample(Network<float>&     dec_dense,
                     ConvNetwork<float>& dec_conv,
                     const MelTransform<float>& mel,
                     const std::string& donor_path,
                     const std::string& out_path,
                     std::size_t count,
                     std::size_t N_patches,
                     unsigned    seed) {
    WavData donor = load_wav(donor_path);
    std::vector<std::vector<float>> lmD, phD;
    stft_decomp(donor.samples, lmD, phD);
    if (phD.empty()) { std::cerr << "donor too short\n"; return 1; }

    std::cout << "sampling " << count << " anchor latent(s) ~ N(0, I)"
              << " (seed=" << seed << "), walking over " << N_patches
              << " patches, donor phase = " << donor_path << "\n";

    Matrix<float> anchors(LATENT, count);
    anchors.randomizeNormal(0.0f, 1.0f, seed);

    // Build z trajectory linearly through anchors across N_patches.
    Matrix<float> Z(LATENT, N_patches);
    for (std::size_t k = 0; k < N_patches; ++k) {
        float u;
        if (count == 1 || N_patches == 1) {
            u = 0.0f;
        } else {
            u = static_cast<float>(k) * (count - 1)
              / static_cast<float>(N_patches - 1);   // 0..count-1
        }
        const std::size_t a    = std::min(static_cast<std::size_t>(u),
                                          count - 1);
        const std::size_t a1   = std::min(a + 1, count - 1);
        const float       frac = u - static_cast<float>(a);
        for (std::size_t d = 0; d < LATENT; ++d)
            Z(d, k) = (1.0f - frac) * anchors(d, a) + frac * anchors(d, a1);
    }

    Tensor4D<float> patches = decode_z(dec_dense, dec_conv, Z);
    auto cols   = patches_to_columns(patches);
    auto frames = build_frames(cols, phD, mel);
    auto out    = istft(frames, FRAME_SIZE, HOP_SIZE, 0);
    write_normalized(out_path, out, donor.sample_rate);
    std::cout << "wrote " << out_path
              << "  (sampling uses donor phase; Griffin-Lim would sharpen)\n";
    return 0;
}

// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n"
                  << "  convae_generate <prefix> interp <A.wav> <B.wav> "
                     "<out.wav> [N_patches]\n"
                  << "  convae_generate <prefix> sample <donor.wav> <out.wav> "
                     "[count] [N_patches] [seed]\n";
        return 1;
    }
    const std::string prefix = argv[1];
    const std::string mode   = argv[2];

    std::cout << "weft :: convae_generate\n";
    ConvNetwork<float> enc_conv, dec_conv;
    Network<float>     enc_dense, dec_dense;
    try {
        load_models(prefix, enc_conv, enc_dense, dec_dense, dec_conv);
    } catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "loaded model: " << prefix
              << ".{enc_conv,enc_dense,dec_dense,dec_conv}\n";

    // Build the mel filterbank used at training time.  Must match
    // train_conv_audio_vae exactly (same n_mels, n_bins, sample_rate,
    // and scale).
    const MelTransform<float> mel(FREQ_BINS, FREQ_BINS_IN, 44100.0f);

    try {
        if (mode == "interp") {
            if (argc < 6) {
                std::cerr << "interp needs <A> <B> <out> [N_patches]\n";
                return 1;
            }
            const std::size_t N_patches = (argc > 6)
                ? std::max(1, std::atoi(argv[6])) : 8;
            return do_interp(enc_conv, enc_dense, dec_dense, dec_conv, mel,
                             argv[3], argv[4], argv[5], N_patches);
        }
        if (mode == "sample") {
            if (argc < 5) {
                std::cerr << "sample needs <donor> <out> [count] [N_patches] "
                             "[seed]\n";
                return 1;
            }
            const std::size_t count = (argc > 5)
                ? std::max(1, std::atoi(argv[5])) : 4;
            const std::size_t N_patches = (argc > 6)
                ? std::max(1, std::atoi(argv[6])) : 8;
            const unsigned seed = (argc > 7)
                ? static_cast<unsigned>(std::atoi(argv[7])) : 1u;
            return do_sample(dec_dense, dec_conv, mel, argv[3], argv[4],
                             count, N_patches, seed);
        }
        std::cerr << "unknown mode: " << mode
                  << " (use interp or sample)\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
