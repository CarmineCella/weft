// vae_remap / vae_remap.cpp
//
// "Remap" an arbitrary input sound (a song, a voice, anything) onto the
// manifold of SOL timbres learned by the audio VAE.  Each STFT frame's
// magnitude spectrum is pushed through the VAE (encode to the latent mean,
// decode), which can only produce SOL-like spectra -- so an out-of-
// distribution frame comes back as its nearest orchestral interpretation.
// Recombined with the INPUT's own phase and inverse-STFT'd, the result is
// the input's rhythm and structure rendered in orchestral timbre.
//
// This is the projection-onto-the-manifold application: autoencoding
// out-of-distribution input through a decoder that only knows SOL.
//
// We reuse the input phase (rather than reconstructing it) because the VAE
// models magnitude only.  That's what makes this the robust, good-sounding
// application -- there is no phase to invent.
//
// The architecture rebuilt here MUST match train_audio_vae (Network::load
// checks shapes and will throw otherwise).
//
// Usage:  vae_remap <model_prefix> <input.wav> <output.wav>
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
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

constexpr std::size_t FRAME_SIZE = 4096;
constexpr std::size_t HOP_SIZE   = 2048;
constexpr std::size_t FEAT_DIM   = FRAME_SIZE / 2 + 1;   // 2049
constexpr std::size_t LATENT     = 32;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: vae_remap <model_prefix> <input.wav> <output.wav>\n";
        return 1;
    }
    const std::string prefix = argv[1];
    const std::string in_path = argv[2];
    const std::string out_path = argv[3];

    std::cout << "weft :: vae_remap\n";

    // ---- Rebuild the architecture and load weights ----
    Network<float> enc;
    enc.add<Dense>(FEAT_DIM, 256);
    enc.add<ReLU>();
    enc.add<Dense>(256, 64);
    enc.add<ReLU>();
    enc.add<Dense>(64, 2 * LATENT);

    Network<float> dec;
    dec.add<Dense>(LATENT, 64);
    dec.add<ReLU>();
    dec.add<Dense>(64, 256);
    dec.add<ReLU>();
    dec.add<Dense>(256, FEAT_DIM);

    try {
        enc.load(prefix + ".enc");
        dec.load(prefix + ".dec");
    } catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << "\n";
        return 1;
    }
    enc.eval(); dec.eval();
    std::cout << "loaded model: " << prefix << ".enc / .dec\n";

    // ---- Load input audio ----
    WavData wav;
    try { wav = load_wav(in_path); }
    catch (const std::exception& e) {
        std::cerr << "error loading audio: " << e.what() << "\n";
        return 1;
    }
    std::cout << "input: " << in_path << "  (" << wav.samples.size()
              << " samples @ " << wav.sample_rate << " Hz)\n";
    if (wav.sample_rate != 44100)
        std::cout << "note: model trained at 44100 Hz; other rates shift the "
                     "frequency-to-bin mapping and may sound off-pitch\n";

    // ---- STFT ----
    auto frames = stft(wav.samples, FRAME_SIZE, HOP_SIZE);
    const std::size_t n = frames.size();
    if (n == 0) {
        std::cerr << "input too short for one frame\n";
        return 1;
    }
    std::cout << "frames: " << n << "\n";

    // ---- Build the log-magnitude matrix (2049 x n) and stash phases ----
    Matrix<float> logmag(FEAT_DIM, n);
    std::vector<std::vector<float>> phases(n);
    for (std::size_t f = 0; f < n; ++f) {
        auto mag = magnitude(frames[f]);          // 2049 bins
        phases[f] = phase(frames[f]);             // reused at synthesis
        for (std::size_t i = 0; i < FEAT_DIM; ++i)
            logmag(i, f) = std::log(1.0f + mag[i]);
    }

    // ---- Encode (mean) -> decode, in one batched pass ----
    std::cout << "projecting onto the SOL manifold...\n";
    Matrix<float> h = enc.forward(logmag);        // (2*LATENT, n)
    Matrix<float> mu(LATENT, n);
    for (std::size_t d = 0; d < LATENT; ++d)
        for (std::size_t f = 0; f < n; ++f)
            mu(d, f) = h(d, f);
    Matrix<float> recon = dec.forward(mu);        // (2049, n) log-magnitudes

    // ---- Rebuild frames: new magnitude + original phase ----
    std::vector<std::vector<std::complex<float>>> out_frames(n);
    for (std::size_t f = 0; f < n; ++f) {
        std::vector<float> new_mag(FEAT_DIM);
        for (std::size_t i = 0; i < FEAT_DIM; ++i) {
            float lm = std::max(0.0f, recon(i, f));    // log-mag >= 0
            new_mag[i] = std::expm1(lm);               // invert log(1 + mag)
            if (new_mag[i] < 0.0f) new_mag[i] = 0.0f;
        }
        out_frames[f] = rebuild_hermitian(new_mag, phases[f]);
    }

    // ---- iSTFT ----
    auto out = istft(out_frames, FRAME_SIZE, HOP_SIZE, wav.samples.size());

    // ---- Suppress edge clicks, normalise robustly, write ----
    //
    // The iSTFT amplifies tiny errors at the edges (where the WOLA
    // synthesis-window power is small but above the divide-by-zero
    // threshold) into an audible click.  A one-hop cosine fade (~46 ms at
    // 44.1 kHz) covers that region; inaudible as a fade.
    {
        const std::size_t fade = std::min<std::size_t>(2048, out.size() / 4);
        for (std::size_t i = 0; i < fade; ++i) {
            const float w = 0.5f * (1.0f - std::cos(3.14159265f * i / fade));
            out[i]                  *= w;
            out[out.size() - 1 - i] *= w;
        }
    }
    // Percentile-based normalisation: plain peak-to-0.9 is hostage to
    // any single transient (residual click, model artifact), which can
    // pull the whole signal down 100x.  The 99th percentile of |x| is
    // robust to outliers; samples that exceed 1.0 after the gain are
    // clipped on write.
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
    try { save_wav(out_path, out, wav.sample_rate); }
    catch (const std::exception& e) {
        std::cerr << "error writing output: " << e.what() << "\n";
        return 1;
    }
    std::cout << "wrote " << out_path << "  (fade + 99th-percentile normalised)\n";
    return 0;
}
