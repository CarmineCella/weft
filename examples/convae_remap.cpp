// convae_remap.cpp
//
// "Remap" an arbitrary input sound onto the manifold of timbres learned
// by the ConvVAE.  Same spirit as vae_remap, but works in 2D patches:
// 16-frame log-magnitude tiles are slid across the input with 50%
// overlap, each tile is encoded to a latent (mu only, no sampling) and
// decoded back to a tile of magnitudes.  Overlap-add the decoded tiles
// in the spectrogram domain, upsample frequency back to 2049 bins, pair
// with the INPUT's own phase, iSTFT, and write.
//
// What this is: an autoencoder-style projection through the learned
// manifold.  An out-of-distribution input (speech, noise, anything not
// in the training corpus) comes back as its nearest interpretation in
// the VAE's timbre vocabulary -- the same way vae_remap projects onto
// the dense VAE's frame manifold, but here the projection respects
// short-time evolution (because the patch is 2D in time-frequency).
//
// Why we keep the input's phase: the VAE models magnitude only, so any
// reconstructed magnitude needs a phase from somewhere.  Reusing the
// input's phase frame-by-frame gives the output the input's rhythm and
// articulation, while the timbre is whatever the VAE projects.
//
// The architecture rebuilt here MUST match train_conv_audio_vae.
//
// Usage:  convae_remap <model_prefix> <input.wav> <output.wav>

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
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

// ---- must match train_conv_audio_vae -----------------------------------
constexpr std::size_t FRAME_SIZE   = 4096;
constexpr std::size_t HOP_SIZE     = 2048;
constexpr std::size_t FREQ_BINS_IN = 2049;
constexpr std::size_t FREQ_BINS    = 64;
constexpr std::size_t TIME_FRAMES  = 16;
constexpr std::size_t LATENT       = 32;
constexpr std::size_t FLAT         = 64 * 8 * 2;

// Stride between consecutive patch starts.  TIME_FRAMES/2 = 50% overlap,
// which gives smoother boundaries than non-overlapping patches at 2x
// the encode/decode work.  Each input frame is covered by exactly 2
// patches (except the very first/last few, handled below).
constexpr std::size_t PATCH_HOP = TIME_FRAMES / 2;   // 8

// --------------------------------------------------------------------------
// Same model rebuilding as convae_generate.  Kept here so the file is
// self-contained and easy to read against vae_remap.
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
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: convae_remap <model_prefix> "
                     "<input.wav> <output.wav>\n";
        return 1;
    }
    const std::string prefix   = argv[1];
    const std::string in_path  = argv[2];
    const std::string out_path = argv[3];

    std::cout << "weft :: convae_remap\n";

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
    // train_conv_audio_vae exactly.
    const MelTransform<float> mel(FREQ_BINS, FREQ_BINS_IN, 44100.0f);

    // ---- load + STFT ------------------------------------------------
    WavData wav;
    try { wav = load_wav(in_path); }
    catch (const std::exception& e) {
        std::cerr << "error loading audio: " << e.what() << "\n"; return 1;
    }
    std::cout << "input: " << in_path << "  (" << wav.samples.size()
              << " samples @ " << wav.sample_rate << " Hz)\n";
    if (wav.sample_rate != 44100)
        std::cout << "note: model trained at 44100 Hz; other rates shift the "
                     "frequency-to-bin mapping and may sound off-pitch\n";

    auto frames = stft(wav.samples, FRAME_SIZE, HOP_SIZE);
    const std::size_t n = frames.size();
    if (n < TIME_FRAMES) {
        std::cerr << "input too short for one " << TIME_FRAMES
                  << "-frame patch\n"; return 1;
    }
    std::cout << "frames: " << n << "\n";

    // ---- log-magnitudes (full 2049) + phases for resynthesis --------
    std::vector<std::vector<float>> phases(n);
    std::vector<std::vector<float>> logmag_hi(n, std::vector<float>(FREQ_BINS_IN));
    for (std::size_t f = 0; f < n; ++f) {
        auto mag = magnitude(frames[f]);
        phases[f] = phase(frames[f]);
        for (std::size_t i = 0; i < FREQ_BINS_IN; ++i)
            logmag_hi[f][i] = std::log1p(mag[i]);
    }

    // ---- plan patch starts ------------------------------------------
    // Every PATCH_HOP frames starting from 0, last one anchored to n.
    // The final patch is forced to end at frame n so we don't leave the
    // tail uncovered (the loop step might overshoot by a frame or two).
    std::vector<std::size_t> starts;
    for (std::size_t s = 0; s + TIME_FRAMES <= n; s += PATCH_HOP)
        starts.push_back(s);
    if (starts.empty() || starts.back() + TIME_FRAMES < n) {
        const std::size_t tail_start = n - TIME_FRAMES;
        if (starts.empty() || starts.back() != tail_start)
            starts.push_back(tail_start);
    }
    const std::size_t P = starts.size();
    std::cout << "patches: " << P << " (hop=" << PATCH_HOP
              << ", overlap=" << (TIME_FRAMES - PATCH_HOP) << " frames)\n";

    // ---- build input tensor (P, 1, FREQ_BINS, TIME_FRAMES) ----------
    Tensor4D<float> Xpatches(P, 1, FREQ_BINS, TIME_FRAMES);
    std::vector<float> col_lo(FREQ_BINS);
    for (std::size_t p = 0; p < P; ++p) {
        const std::size_t s = starts[p];
        for (std::size_t t = 0; t < TIME_FRAMES; ++t) {
            mel.linear_logmag_to_logmel(logmag_hi[s + t].data(), col_lo.data());
            for (std::size_t f = 0; f < FREQ_BINS; ++f)
                Xpatches(p, 0, f, t) = col_lo[f];
        }
    }

    // ---- encode-then-decode all patches at once ---------------------
    std::cout << "projecting through the latent manifold...\n";
    Tensor4D<float> enc_feat = enc_conv.forward(Xpatches);
    Matrix<float>   enc_flat = flatten(enc_feat);
    Matrix<float>   h        = enc_dense.forward(enc_flat);          // (2*LATENT, P)
    Matrix<float>   mu(LATENT, P);
    for (std::size_t d = 0; d < LATENT; ++d)
        for (std::size_t p = 0; p < P; ++p)
            mu(d, p) = h(d, p);
    Matrix<float>   dec_flat  = dec_dense.forward(mu);
    Tensor4D<float> dec_feat  = unflatten(dec_flat, P, 64, 8, 2);
    Tensor4D<float> Ypatches  = dec_conv.forward(dec_feat);          // (P,1,FREQ,TIME)

    // ---- overlap-add patches into output low-res spectrogram --------
    // Rectangular window (uniform weight 1) keeps this simple: each
    // overlapped frame is the mean of the patches that cover it.
    std::vector<std::vector<float>> out_lo(n, std::vector<float>(FREQ_BINS, 0.0f));
    std::vector<int> coverage(n, 0);
    for (std::size_t p = 0; p < P; ++p) {
        const std::size_t s = starts[p];
        for (std::size_t t = 0; t < TIME_FRAMES; ++t) {
            for (std::size_t f = 0; f < FREQ_BINS; ++f)
                out_lo[s + t][f] += Ypatches(p, 0, f, t);
            ++coverage[s + t];
        }
    }
    for (std::size_t f = 0; f < n; ++f) {
        if (coverage[f] == 0) continue;
        const float inv = 1.0f / static_cast<float>(coverage[f]);
        for (std::size_t i = 0; i < FREQ_BINS; ++i)
            out_lo[f][i] *= inv;
    }

    // ---- invert mel to linear mag, rebuild complex frames -----------
    std::vector<std::vector<std::complex<float>>> out_frames(n);
    std::vector<float> mag_col(FREQ_BINS_IN);
    for (std::size_t f = 0; f < n; ++f) {
        if (coverage[f] == 0) {
            // No patch reached this frame -- shouldn't happen given the
            // tail-anchoring above, but as a defence, pass the original
            // log-mag through (linear domain).
            for (std::size_t i = 0; i < FREQ_BINS_IN; ++i)
                mag_col[i] = std::expm1(std::max(0.0f, logmag_hi[f][i]));
        } else {
            mel.logmel_to_linear_mag(out_lo[f].data(), mag_col.data());
        }
        out_frames[f] = rebuild_hermitian(mag_col, phases[f]);
    }

    // ---- iSTFT, edge fade + normalise, write ------------------------
    auto out = istft(out_frames, FRAME_SIZE, HOP_SIZE, wav.samples.size());
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
    try { save_wav(out_path, out, wav.sample_rate); }
    catch (const std::exception& e) {
        std::cerr << "error writing output: " << e.what() << "\n"; return 1;
    }
    std::cout << "wrote " << out_path
              << "  (fade + 99th-percentile normalised)\n";
    return 0;
}
