// train_conv_audio_vae.cpp
//
// Train a CONVOLUTIONAL variational autoencoder on log-magnitude
// spectrogram PATCHES (2D time-frequency tiles, not single frames).
// Reuses the same .feat cache that train_audio_vae consumes -- the
// same data ingestion pipeline, just sliced differently.
//
// Why bother building a conv version when the dense VAE works.  The
// dense VAE encodes each frame independently, so it can capture
// timbre but not time-evolution.  The conv version sees a local
// (freq x time) tile, which lets it model how formants move, how
// attacks differ from sustains, how vibrato unfolds.  In return,
// interpolations in latent space sound smoother and more "musical."
//
// Input: a per_frame logmag .feat file produced by
//     extract_features <wav_dir> <out.feat> logmag per_frame
//
// The cache stores 2049 magnitude bins per frame (from
// FRAME_SIZE=4096 STFTs).  We convert each frame to a 64-bin log-mel
// representation via a triangular mel filterbank -- mel-spacing
// preserves harmonic peaks in the low/mid kHz range where music
// lives, much better than linear block-averaging would.
// Each TIME_FRAMES=16 consecutive frames from the same file becomes
// one (1, 64, 16) training patch.  At HOP=2048 samples that's ~0.8 s
// of audio per patch.
//
// Architecture:
//   Encoder:
//     ConvNetwork  Conv2D(1->16) -> ReLU4D -> MaxPool(2)    (1,64,16) -> (16,32,8)
//                  Conv2D(16->32) -> ReLU4D -> MaxPool(2)              -> (32,16,4)
//                  Conv2D(32->64) -> ReLU4D -> MaxPool(2)              -> (64, 8,2)
//                  -- flatten --                                      -> 2048
//     Network      Dense(1024, 256) -> ReLU -> Dense(256, 2*LATENT)
//   Reparam:
//     z = mu + sigma * eps,  eps ~ N(0, I)
//   Decoder mirrors:
//     Network      Dense(LATENT, 256) -> ReLU -> Dense(256, 1024) -> ReLU
//                  -- unflatten --
//     ConvNetwork  Upsample(2) -> Conv2D(64->32) -> ReLU4D
//                  Upsample(2) -> Conv2D(32->16) -> ReLU4D
//                  Upsample(2) -> Conv2D(16->1)               (linear output)
//
// Linear output (no sigmoid): log-magnitudes are non-negative and
// unbounded.  The generate/remap apps clamp to >= 0 before
// exp-inverting to magnitude.
//
// Usage:
//   train_conv_audio_vae <feature.feat> <model_prefix> [epochs] [window_hop]
//
// Writes <prefix>.enc_conv  <prefix>.enc_dense
//        <prefix>.dec_dense <prefix>.dec_conv

#include "Adam.h"
#include "Conv2D.h"
#include "ConvNetwork.h"
#include "Data.h"
#include "Dense.h"
#include "FeatureCache.h"
#include "Flatten.h"
#include "Matrix.h"
#include "MaxPool2D.h"
#include "MSE.h"
#include "Network.h"
#include "ReLU.h"
#include "ReLU4D.h"
#include "Tensor4D.h"
#include "MelTransform.h"
#include "Upsample2D.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace weft;

// ---- patch shape ---------------------------------------------------------
constexpr std::size_t FREQ_BINS_IN  = 2049;   // from FRAME_SIZE=4096 in extract_features
constexpr std::size_t FREQ_BINS     = 64;     // after block-averaging
constexpr std::size_t TIME_FRAMES   = 16;     // 16 frames * ~46ms hop = ~0.8s patches.
                                              // Small enough that most TinySOL notes
                                              // produce >=1 patch; larger windows
                                              // would exclude shorter notes.

// ---- VAE hyperparameters -------------------------------------------------
constexpr std::size_t LATENT = 32;
constexpr float       BETA   = 1.0f;
constexpr std::size_t FLAT   = 64 * 8 * 2;    // 1024 (= 64 channels x 8 freq x 2 time at bottleneck)

// --------------------------------------------------------------------------
// Frequency-axis transform.  We compress the 2049 linear bins from the
// STFT down to FREQ_BINS mel bins via the standard triangular filterbank
// in MelTransform.  Mel scaling preserves the harmonic structure that
// makes audio sound like audio (more bins in low/mid where harmonics
// live) much better than linear block-averaging would.
// --------------------------------------------------------------------------
// (filterbank is built once in main(), passed into patches_from_cache)

// --------------------------------------------------------------------------
// Pull all (FREQ_BINS, TIME_FRAMES) patches from the cache.  Walks the
// columns in order, groups them by file_id (which is sequential within
// extract_features' output), then slides a window of TIME_FRAMES with
// step window_hop across each group.  Each patch ends up as one row of
// patches.{N=count, 1, FREQ_BINS, TIME_FRAMES}.
// --------------------------------------------------------------------------
static Tensor4D<float>
patches_from_cache(const CachedFeatures& cache, const MelTransform<float>& mel,
                   std::size_t window_hop) {
    if (cache.X.rows() != FREQ_BINS_IN)
        throw std::runtime_error(
            "expected " + std::to_string(FREQ_BINS_IN) +
            " freq bins from FRAME_SIZE=4096 logmag features, got " +
            std::to_string(cache.X.rows()));
    if (!cache.per_frame)
        throw std::runtime_error("expected per_frame=true cache");
    if (cache.feature_type != "logmag")
        throw std::runtime_error("expected logmag features");

    const std::size_t N = cache.X.cols();
    // First pass: find file boundaries.
    std::vector<std::pair<std::size_t, std::size_t>> runs;   // [start, end)
    std::size_t run_start = 0;
    for (std::size_t j = 1; j < N; ++j) {
        if (cache.file_ids[j] != cache.file_ids[run_start]) {
            runs.emplace_back(run_start, j);
            run_start = j;
        }
    }
    runs.emplace_back(run_start, N);

    // Second pass: count patches.
    std::size_t n_patches = 0;
    for (auto [s, e] : runs) {
        const std::size_t len = e - s;
        if (len < TIME_FRAMES) continue;
        n_patches += (len - TIME_FRAMES) / window_hop + 1;
    }
    if (n_patches == 0)
        throw std::runtime_error(
            "no files have at least " + std::to_string(TIME_FRAMES) +
            " frames; reduce TIME_FRAMES or check the feature file");

    // Third pass: allocate and fill.
    Tensor4D<float> P(n_patches, 1, FREQ_BINS, TIME_FRAMES);
    std::size_t out_idx = 0;
    std::vector<float> col_lores(FREQ_BINS);   // scratch buffer for one downsampled column
    for (auto [s, e] : runs) {
        const std::size_t len = e - s;
        if (len < TIME_FRAMES) continue;
        for (std::size_t t0 = 0; t0 + TIME_FRAMES <= len; t0 += window_hop) {
            for (std::size_t t = 0; t < TIME_FRAMES; ++t) {
                // Read column s + t0 + t into the downsampled buffer.
                // Matrix is (FREQ_BINS_IN x N) row-major; column access via
                // operator() is element-wise.  One temp buffer copy keeps
                // the inner block-average loop cache-friendly.
                std::vector<float> col_hi(FREQ_BINS_IN);
                for (std::size_t f = 0; f < FREQ_BINS_IN; ++f)
                    col_hi[f] = cache.X(f, s + t0 + t);
                mel.linear_logmag_to_logmel(col_hi.data(), col_lores.data());
                for (std::size_t f = 0; f < FREQ_BINS; ++f)
                    P(out_idx, 0, f, t) = col_lores[f];
            }
            ++out_idx;
        }
    }
    return P;
}

// Select a batch of patches by index.
static Tensor4D<float>
select_patches(const Tensor4D<float>& X, const std::vector<std::size_t>& idx) {
    const std::size_t patch = FREQ_BINS * TIME_FRAMES;
    Tensor4D<float> out(idx.size(), 1, FREQ_BINS, TIME_FRAMES);
    for (std::size_t i = 0; i < idx.size(); ++i)
        std::memcpy(out.data() + i * patch,
                    X.data() + idx[i] * patch,
                    patch * sizeof(float));
    return out;
}

// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    if (argc < 3) {
        std::cerr << "usage: train_conv_audio_vae <feature.feat> "
                     "<model_prefix> [epochs] [window_hop]\n"
                     "  feature file must be per_frame logmag, produced by:\n"
                     "    extract_features <wav_dir> <out.feat> logmag per_frame\n"
                     "  window_hop: frames between consecutive patches from\n"
                     "              one file.  Default 16 (50%% overlap of\n"
                     "              the 32-frame window).\n";
        return 1;
    }
    const std::string feature_path = argv[1];
    const std::string prefix       = argv[2];
    const int         epochs       = (argc > 3) ? std::atoi(argv[3]) : 20;
    const std::size_t window_hop   =
        (argc > 4) ? std::max(1, std::atoi(argv[4])) : TIME_FRAMES / 2;
    constexpr int CKPT_EVERY = 5;

    std::cout << "weft :: train conv audio VAE\n"
              << "  feature file: " << feature_path << "\n"
              << "  prefix:       " << prefix       << "\n"
              << "  epochs:       " << epochs       << "\n"
              << "  window hop:   " << window_hop   << " frames\n"
              << "  patch shape:  1 x " << FREQ_BINS << " x "
              << TIME_FRAMES
              << "  (mel filterbank from 2049 linear bins,\n"
              << "                 ~0.8s audio per patch)\n"
              << "  latent:       " << LATENT << "\n\n";

    // ---- 1. Load and slice ------------------------------------------
    CachedFeatures cache;
    try {
        cache = load_features(feature_path);
    } catch (const std::exception& e) {
        std::cerr << "error loading features: " << e.what() << "\n";
        return 1;
    }
    std::cout << "loaded:  " << cache.X.cols() << " frames, "
              << cache.class_names.size() << " classes ("
              << cache.feature_type << ", "
              << (cache.per_frame ? "per_frame" : "averaged") << ")\n";

    auto t_start = clock::now();
    Tensor4D<float> X;
    try {
        // Mel filterbank: 64 mel bins from 2049 linear bins at 44100 Hz.
        // Trained model bakes in this choice; convae_generate/convae_remap
        // must build the same filterbank to invert correctly.
        const MelTransform<float> mel(FREQ_BINS, FREQ_BINS_IN, 44100.0f);
        X = patches_from_cache(cache, mel, window_hop);
    } catch (const std::exception& e) {
        std::cerr << "error slicing patches: " << e.what() << "\n";
        return 1;
    }
    const std::size_t N = X.N();
    double slice_time = std::chrono::duration<double>(clock::now() - t_start).count();
    std::cout << "patches: " << N << " ("
              << (N * FREQ_BINS * TIME_FRAMES * 4) / (1024 * 1024) << " MB, "
              << std::fixed << std::setprecision(1) << slice_time << "s)\n\n";

    // ---- 2. Build encoder + decoder ---------------------------------
    ConvNetwork<float> enc_conv;
    enc_conv.add<Conv2D>  (1,  16, 3, 1, 1);
    enc_conv.add<ReLU4D>  ();
    enc_conv.add<MaxPool2D>(2);
    enc_conv.add<Conv2D>  (16, 32, 3, 1, 1);
    enc_conv.add<ReLU4D>  ();
    enc_conv.add<MaxPool2D>(2);
    enc_conv.add<Conv2D>  (32, 64, 3, 1, 1);
    enc_conv.add<ReLU4D>  ();
    enc_conv.add<MaxPool2D>(2);

    Network<float> enc_dense;
    enc_dense.add<Dense>(FLAT, 256);
    enc_dense.add<ReLU>();
    enc_dense.add<Dense>(256, 2 * LATENT);

    Network<float> dec_dense;
    dec_dense.add<Dense>(LATENT, 256);
    dec_dense.add<ReLU>();
    dec_dense.add<Dense>(256, FLAT);
    dec_dense.add<ReLU>();

    ConvNetwork<float> dec_conv;
    dec_conv.add<Upsample2D>(2);
    dec_conv.add<Conv2D>    (64, 32, 3, 1, 1);
    dec_conv.add<ReLU4D>    ();
    dec_conv.add<Upsample2D>(2);
    dec_conv.add<Conv2D>    (32, 16, 3, 1, 1);
    dec_conv.add<ReLU4D>    ();
    dec_conv.add<Upsample2D>(2);
    dec_conv.add<Conv2D>    (16,  1, 3, 1, 1);   // linear output

    std::cout << "encoder (conv):\n" << enc_conv.summary()
              << "  -- flatten --\n"
              << "encoder (dense):\n" << enc_dense.summary()
              << "\nz = mu + sigma * eps, eps ~ N(0, I)\n\n"
              << "decoder (dense):\n" << dec_dense.summary()
              << "  -- unflatten --\n"
              << "decoder (conv):\n" << dec_conv.summary()
              << "loss: MSE + " << BETA << " * KL\n\n";

    MSE<float>  mse;
    Adam<float> opt(1e-3f);

    const std::size_t batch_size = 32;

    auto save_models = [&](const std::string& tag) {
        try {
            enc_conv .save(prefix + ".enc_conv");
            enc_dense.save(prefix + ".enc_dense");
            dec_dense.save(prefix + ".dec_dense");
            dec_conv .save(prefix + ".dec_conv");
            std::cout << "  [" << tag << " saved -> " << prefix
                      << ".{enc_conv,enc_dense,dec_dense,dec_conv}]\n"
                      << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "  save failed: " << e.what() << "\n";
        }
    };

    // ---- 3. Training loop -------------------------------------------
    std::cout << "epoch  recon       KL        total      time\n";
    std::cout << "-----  ---------   -------   ---------   -----\n";

    unsigned eps_seed = 1234;
    auto t_overall = clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_epoch = clock::now();
        enc_conv.train(); enc_dense.train();
        dec_dense.train(); dec_conv.train();

        auto perm = shuffled_indices(N, static_cast<unsigned>(epoch));

        double sum_recon = 0, sum_kl = 0;
        std::size_t n_seen = 0;

        for (std::size_t bs = 0; bs < N; bs += batch_size) {
            const std::size_t be = std::min(N, bs + batch_size);
            std::vector<std::size_t> idx(perm.begin() + bs, perm.begin() + be);
            const std::size_t B = idx.size();

            Tensor4D<float> Xb = select_patches(X, idx);

            // encode: conv -> flatten -> dense -> [mu; logvar]
            Tensor4D<float> feat = enc_conv.forward(Xb);
            Matrix<float>   flat = flatten(feat);
            Matrix<float>   h    = enc_dense.forward(flat);

            Matrix<float> mu(LATENT, B), logvar(LATENT, B);
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    mu(d, j)     = h(d, j);
                    logvar(d, j) = h(LATENT + d, j);
                }

            Matrix<float> sigma = logvar.apply([](float v){
                return std::exp(0.5f * v);
            });
            Matrix<float> eps(LATENT, B);
            eps.randomizeNormal(0.0f, 1.0f, eps_seed++);
            Matrix<float> z = mu + hadamard(sigma, eps);

            // decode: dense -> unflatten -> conv
            Matrix<float>   dec_flat = dec_dense.forward(z);
            Tensor4D<float> dec_feat = unflatten(dec_flat, B, 64, 8, 2);
            Tensor4D<float> recon    = dec_conv.forward(dec_feat);

            Matrix<float> recon_flat = flatten(recon);
            Matrix<float> input_flat = flatten(Xb);
            const float   L          = mse.forward(recon_flat, input_flat);

            // backward
            Matrix<float>   dRecon_flat = mse.backward();
            Tensor4D<float> dRecon = unflatten(dRecon_flat, B, 1,
                                               FREQ_BINS, TIME_FRAMES);
            Tensor4D<float> dDecFeat = dec_conv.backward(dRecon);
            Matrix<float>   dDecFlat = flatten(dDecFeat);
            Matrix<float>   dz       = dec_dense.backward(dDecFlat);

            // KL value (reporting)
            double kl = 0;
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    const float s2 = sigma(d, j) * sigma(d, j);
                    kl += -0.5 * (1.0 + logvar(d, j)
                                       - mu(d, j) * mu(d, j) - s2);
                }
            const float L_kl = static_cast<float>(kl / B);

            // gradient into encoder outputs (recon + KL paths)
            const float invB = 1.0f / static_cast<float>(B);
            Matrix<float> dh(2 * LATENT, B);
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    const float s = sigma(d, j);
                    const float gmu = dz(d, j) + BETA * invB * mu(d, j);
                    const float glv = dz(d, j) * eps(d, j) * 0.5f * s
                                    + BETA * invB * 0.5f * (s * s - 1.0f);
                    dh(d, j)          = gmu;
                    dh(LATENT + d, j) = glv;
                }

            Matrix<float>   dEncFlat = enc_dense.backward(dh);
            Tensor4D<float> dEncFeat = unflatten(dEncFlat, B, 64, 8, 2);
            (void)enc_conv.backward(dEncFeat);

            dec_conv .update(opt);
            dec_dense.update(opt);
            enc_dense.update(opt);
            enc_conv .update(opt);

            sum_recon += static_cast<double>(L)    * B;
            sum_kl    += static_cast<double>(L_kl) * B;
            n_seen    += B;
        }

        const float recon = static_cast<float>(sum_recon / n_seen);
        const float kl    = static_cast<float>(sum_kl    / n_seen);
        double secs = std::chrono::duration<double>(
                          clock::now() - t_epoch).count();
        std::cout << std::fixed
                  << std::setw(5) << epoch << "  "
                  << std::setprecision(4) << std::setw(9) << recon << "   "
                  << std::setprecision(4) << std::setw(7) << kl << "   "
                  << std::setprecision(4) << std::setw(9)
                  << (recon + BETA * kl) << "   "
                  << std::setprecision(1) << std::setw(5) << secs << "s\n"
                  << std::flush;

        if (epoch % CKPT_EVERY == 0 && epoch != epochs)
            save_models("checkpoint @ epoch " + std::to_string(epoch));
    }

    double total = std::chrono::duration<double>(
                       clock::now() - t_overall).count();
    std::cout << "\ntotal training time: " << std::fixed
              << std::setprecision(1) << total << "s\n";
    save_models("final");
    std::cout << "\nlatent dim: " << LATENT
              << "  (convae_generate / convae_remap must rebuild this same\n"
              << "   architecture before loading the four model files)\n";
    return 0;
}
