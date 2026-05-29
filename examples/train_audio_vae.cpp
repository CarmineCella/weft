// train_audio_vae / train_audio_vae.cpp
//
// Train a variational autoencoder on per-frame log-magnitude spectra of
// the SOL sounds, and save the trained encoder and decoder so the
// application tools (vae_remap, vae_generate) can load them without
// retraining.  Same train-once / iterate-fast split as the feature cache.
//
// Input is a per-frame logmag feature cache produced by:
//     extract_features <wav_dir> sol_logmag_frames.feat logmag per_frame
//
// Each column of the cache is one frame's log-magnitude spectrum
// (log(1 + |X|), 2049 bins for a 4096-point FFT).  The VAE is exactly the
// MNIST one with bigger layers and a 32-D latent: encoder emits mu and
// logvar, we sample z = mu + sigma*eps, the decoder reconstructs the
// spectrum, and the loss is MSE + beta*KL.  Labels in the cache are
// ignored -- this is unsupervised.
//
// The decoder output is LINEAR (no sigmoid): log-magnitudes are
// non-negative and unbounded, not [0, 1] like pixels, so there's nothing
// to squash.  The apps clamp to >= 0 before exp-inverting to magnitude.
//
// Usage:
//   train_audio_vae <feature.feat> <model_prefix> [epochs]
// Writes:
//   <model_prefix>.enc , <model_prefix>.dec
//
#include "Adam.h"
#include "Data.h"
#include "Dense.h"
#include "FeatureCache.h"
#include "MSE.h"
#include "Network.h"
#include "ReLU.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

constexpr std::size_t LATENT = 32;
constexpr float       BETA   = 1.0f;

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;

    if (argc < 3) {
        std::cerr << "usage: train_audio_vae <feature.feat> <model_prefix> "
                     "[epochs] [subsample]\n"
                     "  subsample N: train on every Nth frame (default 1 = all). "
                     "Adjacent STFT frames are highly redundant, so 2-4 cuts "
                     "epoch time with little quality loss.\n";
        return 1;
    }
    const std::string feature_path = argv[1];
    const std::string prefix       = argv[2];
    const int         epochs       = (argc > 3) ? std::atoi(argv[3]) : 20;
    const std::size_t subsample    =
        (argc > 4) ? std::max(1, std::atoi(argv[4])) : 1;
    constexpr int CKPT_EVERY = 5;   // save a checkpoint every N epochs

    std::cout << "weft :: train audio VAE\n";
    std::cout << "feature file: " << feature_path << "\n";

    CachedFeatures cache;
    try {
        cache = load_features(feature_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const std::size_t FEAT_DIM = cache.X.rows();
    const std::size_t N        = cache.X.cols();

    std::cout << "feature type: " << cache.feature_type
              << " (" << FEAT_DIM << " dim)\n";
    std::cout << "mode:         " << (cache.per_frame ? "per_frame" : "averaged") << "\n";
    std::cout << "frames:       " << N << "\n";
    if (cache.feature_type != "logmag")
        std::cout << "WARNING: expected 'logmag' features for audio reconstruction\n";
    if (!cache.per_frame)
        std::cout << "WARNING: expected per-frame features\n";
    std::cout << "\n";

    // ---- Encoder (emits [mu; logvar]) and decoder ----
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
    dec.add<Dense>(256, FEAT_DIM);     // linear output (log-magnitudes)

    MSE<float>  mse;
    Adam<float> opt(1e-3f);

    const std::size_t batch_size = 256;

    // Subsample the frames: train on every `subsample`-th column.  We
    // subsample the INDEX LIST, not the matrix, so there's no second
    // multi-gigabyte copy -- batches are still selected from cache.X.
    std::vector<std::size_t> base_idx;
    base_idx.reserve(N / subsample + 1);
    for (std::size_t i = 0; i < N; i += subsample) base_idx.push_back(i);
    const std::size_t n_train = base_idx.size();

    std::cout << "architecture:\n"
              << "  encoder (last layer = mu[" << LATENT << "] ++ logvar["
              << LATENT << "]):\n" << enc.summary() << "\n"
              << "  sample: z = mu + sigma * eps,  eps ~ N(0, I)\n"
              << "  decoder:\n" << dec.summary() << "\n"
              << "loss:        MSE + " << BETA << " * KL\n"
              << "optimiser:   Adam (lr=1e-3)\n"
              << "batch:       " << batch_size << "\n"
              << "subsample:   every " << subsample << " frame(s)  -> "
              << n_train << " of " << N << " frames\n"
              << "epochs:      " << epochs
              << "  (checkpoint every " << CKPT_EVERY << ")\n\n";

    std::cout << "epoch  recon       KL        total      time\n";
    std::cout << "-----  ---------   -------   ---------   -----\n";

    auto t_start = clock::now();
    unsigned eps_seed = 1234;

    // Helper: write the current encoder/decoder to <prefix>.enc/.dec.
    auto save_models = [&](const std::string& tag) {
        try {
            enc.save(prefix + ".enc");
            dec.save(prefix + ".dec");
            std::cout << "  [" << tag << " saved -> " << prefix
                      << ".enc/.dec]\n" << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "  save failed: " << e.what() << "\n";
        }
    };

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_epoch = clock::now();
        enc.train(); dec.train();
        // Shuffle the subsampled index list for this epoch.
        std::vector<std::size_t> idx = base_idx;
        {
            auto perm = shuffled_indices(n_train, /*seed=*/epoch);
            std::vector<std::size_t> shuffled(n_train);
            for (std::size_t i = 0; i < n_train; ++i) shuffled[i] = idx[perm[i]];
            idx.swap(shuffled);
        }

        double sum_recon = 0, sum_kl = 0;
        std::size_t n_seen = 0;

        for (std::size_t start = 0; start < n_train; start += batch_size) {
            const std::size_t end = std::min(start + batch_size, n_train);
            std::vector<std::size_t> bidx(idx.begin() + start, idx.begin() + end);
            const std::size_t B = bidx.size();
            Matrix<float> Xb = cache.X.selectColumns(bidx);

            // encode -> split -> reparameterise -> decode
            Matrix<float> h = enc.forward(Xb);
            Matrix<float> mu(LATENT, B), logvar(LATENT, B);
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    mu(d, j)     = h(d, j);
                    logvar(d, j) = h(LATENT + d, j);
                }
            Matrix<float> sigma = logvar.apply([](float v){ return std::exp(0.5f * v); });
            Matrix<float> eps(LATENT, B);
            eps.randomizeNormal(0.0f, 1.0f, eps_seed++);
            Matrix<float> z = mu + hadamard(sigma, eps);

            Matrix<float> recon  = dec.forward(z);
            const float   L      = mse.forward(recon, Xb);
            Matrix<float> dRecon = mse.backward();
            Matrix<float> dz     = dec.backward(dRecon);

            // KL value (reporting)
            double kl = 0;
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    const float s2 = sigma(d, j) * sigma(d, j);
                    kl += -0.5 * (1.0 + logvar(d, j) - mu(d, j) * mu(d, j) - s2);
                }
            const float L_kl = static_cast<float>(kl / B);

            // gradients into encoder outputs (recon path + KL path)
            const float invB = 1.0f / static_cast<float>(B);
            Matrix<float> dh(2 * LATENT, B);
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    const float s = sigma(d, j);
                    const float gmu  = dz(d, j) + BETA * invB * mu(d, j);
                    const float glv  = dz(d, j) * eps(d, j) * 0.5f * s
                                     + BETA * invB * 0.5f * (s * s - 1.0f);
                    dh(d, j)          = gmu;
                    dh(LATENT + d, j) = glv;
                }
            enc.backward(dh);

            dec.update(opt);
            enc.update(opt);

            sum_recon += static_cast<double>(L)    * B;
            sum_kl    += static_cast<double>(L_kl) * B;
            n_seen    += B;
        }

        const float recon = static_cast<float>(sum_recon / n_seen);
        const float kl    = static_cast<float>(sum_kl    / n_seen);
        double secs = std::chrono::duration<double>(clock::now() - t_epoch).count();
        std::cout << std::fixed
                  << std::setw(5) << epoch << "  "
                  << std::setprecision(4) << std::setw(9) << recon << "   "
                  << std::setprecision(4) << std::setw(7) << kl << "   "
                  << std::setprecision(4) << std::setw(9) << (recon + BETA * kl) << "   "
                  << std::setprecision(1) << std::setw(5) << secs << "s\n" << std::flush;

        // Checkpoint periodically (and on the final epoch) so a long run is
        // never all-or-nothing: kill it any time and the latest weights are
        // already on disk at <prefix>.enc/.dec.
        if (epoch % CKPT_EVERY == 0 && epoch != epochs)
            save_models("checkpoint @ epoch " + std::to_string(epoch));
    }

    double total = std::chrono::duration<double>(clock::now() - t_start).count();
    std::cout << "\ntotal training time: " << std::setprecision(1) << total << "s\n";

    // ---- Final save ----
    save_models("final");
    std::cout << "\nlatent dim: " << LATENT
              << "  (apps must rebuild this same architecture before loading)\n";
    return 0;
}
