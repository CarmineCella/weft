// mnist_vae / mnist_vae.cpp
//
// A variational autoencoder on MNIST.  Where the plain autoencoder mapped
// each image to a single latent POINT, the VAE maps it to a latent
// DISTRIBUTION -- a Gaussian with mean mu and variance sigma^2 -- and a
// second loss term pulls every such Gaussian toward the standard normal
// N(0, I).  The payoff: the latent space ends up gap-free, so sampling a
// random z ~ N(0, I) and decoding it produces a plausible new digit.
// That's generation, not just reconstruction.
//
// Three new ideas over the plain AE:
//
//   1. The encoder outputs TWO vectors, mu and logvar, instead of one
//      code.  We have it emit 2*LATENT rows and split them.  (We predict
//      log-variance rather than variance so the network's output is
//      unconstrained -- any real number is a valid logvar, whereas a
//      variance would have to be forced positive.)
//
//   2. Reparameterisation trick.  We need to sample z ~ N(mu, sigma^2)
//      but still backprop through it.  Trick: draw eps ~ N(0, I) as an
//      INPUT and compute z = mu + sigma * eps.  The randomness now enters
//      through eps, which has no parameters, so gradients flow cleanly
//      through mu and sigma.
//
//   3. KL divergence loss.  For a Gaussian posterior against an N(0, I)
//      prior there's a closed form:
//          KL = -0.5 * sum(1 + logvar - mu^2 - sigma^2)
//      The total loss is reconstruction (MSE) + beta * KL.
//
// Gradients pushed back into the encoder (per element; the 1/B batch
// factor matches MSE's convention, so recon and KL terms compose):
//
//   from reconstruction (dz = dL_recon/dz, already 1/B-averaged):
//       dL_recon/dmu     = dz
//       dL_recon/dlogvar = dz * eps * 0.5 * sigma     (chain through sigma)
//   from KL (averaged over batch, hence the 1/B):
//       dL_kl/dmu        = (1/B) * mu
//       dL_kl/dlogvar    = (1/B) * 0.5 * (sigma^2 - 1)
//
// Output BMPs (current directory):
//   vae_reconstructions.bmp  -- originals vs reconstructions (using mu)
//   vae_latent_grid.bmp      -- decode a grid of z over the 2D plane:
//                               the whole digit manifold, laid out
//                               continuously.  THE demo.
//   vae_samples.bmp          -- digits decoded from z ~ N(0, I)
//
// Usage:  mnist_vae [data_dir]   (default ../data/mnist)
//
#include "Adam.h"
#include "Bmp.h"
#include "Data.h"
#include "Dense.h"
#include "MNIST.h"
#include "MSE.h"
#include "Network.h"
#include "ReLU.h"
#include "Sigmoid.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

constexpr std::size_t LATENT = 2;       // 2D so we can render the manifold grid
                                        // (Demo 2).  Pedagogically the clearest
                                        // choice; set to 16 for sharper recon at
                                        // the cost of the grid visualisation.
constexpr float       BETA   = 1.0f;    // weight on the KL term (lower => sharper
                                        // recon but a looser latent; try 0.5)

// Draw a 28x28 image (column `col` of a 784xN matrix, values in [0,1])
// into `bmp` at top-left (x0, y0), scaled up by an integer factor.
static void blit_digit(Bitmap& bmp, const Matrix<float>& M, std::size_t col,
                       int x0, int y0, int scale) {
    for (int r = 0; r < 28; ++r)
        for (int c = 0; c < 28; ++c) {
            float v = M(static_cast<std::size_t>(r * 28 + c), col);
            v = std::max(0.0f, std::min(1.0f, v));
            auto g = static_cast<std::uint8_t>(v * 255.0f + 0.5f);
            for (int dy = 0; dy < scale; ++dy)
                for (int dx = 0; dx < scale; ++dx)
                    bmp.set_gray(x0 + c * scale + dx, y0 + r * scale + dy, g);
        }
}

// Column index of the first image whose label == digit.
static std::size_t find_digit(const std::vector<int>& labels, int digit) {
    for (std::size_t i = 0; i < labels.size(); ++i)
        if (labels[i] == digit) return i;
    return 0;
}

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    const std::string data_dir = (argc > 1) ? argv[1] : "../data/mnist";

    std::cout << "weft :: MNIST variational autoencoder\n";
    std::cout << "data dir: " << data_dir << "\n\n";

    // ---- Load MNIST (labels only used to pick digits for the morph demo) ----
    Matrix<float>    X_train, X_test;
    std::vector<int> y_test;
    try {
        X_train = mnist::load_images<float>(data_dir + "/train-images-idx3-ubyte");
        X_test  = mnist::load_images<float>(data_dir + "/t10k-images-idx3-ubyte");
        y_test  = mnist::load_labels       (data_dir + "/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "error loading MNIST: " << e.what() << "\n";
        std::cerr << "did you run data/download_mnist.sh ?\n";
        return 1;
    }
    std::cout << "loaded " << X_train.cols() << " train, "
              << X_test.cols() << " test images\n\n";

    // ---- Encoder (emits [mu; logvar]) and decoder ----
    Network<float> enc;
    enc.add<Dense>(784, 128);
    enc.add<ReLU>();
    enc.add<Dense>(128, 32);
    enc.add<ReLU>();
    enc.add<Dense>(32, 2 * LATENT);     // top LATENT rows = mu, bottom = logvar

    Network<float> dec;
    dec.add<Dense>(LATENT, 32);
    dec.add<ReLU>();
    dec.add<Dense>(32, 128);
    dec.add<ReLU>();
    dec.add<Dense>(128, 784);
    dec.add<Sigmoid>();

    MSE<float>  mse;
    Adam<float> opt(1e-3f);

    const std::size_t batch_size = 128;
    const int         epochs     = 40;

    std::cout << "architecture:\n"
              << "  encoder (last layer = mu[" << LATENT << "] ++ logvar["
              << LATENT << "]):\n" << enc.summary() << "\n"
              << "  sample: z = mu + sigma * eps,  eps ~ N(0, I)\n"
              << "  decoder:\n" << dec.summary() << "\n"
              << "loss:        MSE + " << BETA << " * KL\n"
              << "optimiser:   Adam (lr=1e-3)\n"
              << "batch:       " << batch_size << "\n"
              << "epochs:      " << epochs << "\n\n";

    std::cout << "epoch  recon      KL        total     time\n";
    std::cout << "-----  --------   -------   --------   -----\n";

    auto t_start = clock::now();
    unsigned eps_seed = 12345;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_epoch = clock::now();
        enc.train(); dec.train();
        auto idx = shuffled_indices(X_train.cols(), /*seed=*/epoch);

        double sum_recon = 0, sum_kl = 0;
        std::size_t n_seen = 0;

        for (std::size_t start = 0; start < X_train.cols(); start += batch_size) {
            const std::size_t end = std::min(start + batch_size, X_train.cols());
            std::vector<std::size_t> bidx(idx.begin() + start, idx.begin() + end);
            const std::size_t B = bidx.size();
            Matrix<float> Xb = X_train.selectColumns(bidx);

            // ---- Forward: encode, split, reparameterise, decode ----
            Matrix<float> h = enc.forward(Xb);          // (2*LATENT, B)

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

            Matrix<float> recon = dec.forward(z);

            // ---- Reconstruction loss + gradient back to z ----
            const float L_recon = mse.forward(recon, Xb);
            Matrix<float> dRecon = mse.backward();      // (784, B), already /B
            Matrix<float> dz = dec.backward(dRecon);    // (LATENT, B) = dL_recon/dz

            // ---- KL value (for reporting) ----
            double kl = 0;
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    const float s2 = sigma(d, j) * sigma(d, j);
                    kl += -0.5 * (1.0 + logvar(d, j) - mu(d, j) * mu(d, j) - s2);
                }
            const float L_kl = static_cast<float>(kl / B);

            // ---- Gradients into the encoder outputs ----
            const float invB = 1.0f / static_cast<float>(B);
            Matrix<float> grad_mu(LATENT, B), grad_logvar(LATENT, B);
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    const float s  = sigma(d, j);
                    // recon path
                    const float gmu_recon = dz(d, j);                       // dz/dmu = 1
                    const float glv_recon = dz(d, j) * eps(d, j) * 0.5f * s; // chain through sigma
                    // KL path (averaged over batch)
                    const float gmu_kl = invB * mu(d, j);
                    const float glv_kl = invB * 0.5f * (s * s - 1.0f);
                    grad_mu(d, j)     = gmu_recon + BETA * gmu_kl;
                    grad_logvar(d, j) = glv_recon + BETA * glv_kl;
                }

            // Reassemble (2*LATENT, B) gradient and backprop the encoder.
            Matrix<float> dh(2 * LATENT, B);
            for (std::size_t d = 0; d < LATENT; ++d)
                for (std::size_t j = 0; j < B; ++j) {
                    dh(d, j)          = grad_mu(d, j);
                    dh(LATENT + d, j) = grad_logvar(d, j);
                }
            enc.backward(dh);

            dec.update(opt);
            enc.update(opt);

            sum_recon += static_cast<double>(L_recon) * B;
            sum_kl    += static_cast<double>(L_kl)    * B;
            n_seen    += B;
        }

        const float recon = static_cast<float>(sum_recon / n_seen);
        const float kl    = static_cast<float>(sum_kl    / n_seen);
        double secs = std::chrono::duration<double>(clock::now() - t_epoch).count();
        std::cout << std::fixed
                  << std::setw(5) << epoch << "  "
                  << std::setprecision(4) << std::setw(8) << recon << "   "
                  << std::setprecision(4) << std::setw(7) << kl << "   "
                  << std::setprecision(4) << std::setw(8) << (recon + BETA * kl) << "   "
                  << std::setprecision(1) << std::setw(4) << secs << "s\n" << std::flush;
    }

    double total = std::chrono::duration<double>(clock::now() - t_start).count();
    std::cout << "\ntotal training time: " << std::setprecision(1) << total << "s\n";

    enc.eval(); dec.eval();

    // Encode helper: returns mu (the distribution mean) for a single image.
    auto encode_mu = [&](const Matrix<float>& X, std::size_t col) {
        Matrix<float> x = X.selectColumns(std::vector<std::size_t>{col});
        Matrix<float> h = enc.forward(x);
        Matrix<float> mu(LATENT, 1);
        for (std::size_t d = 0; d < LATENT; ++d) mu(d, 0) = h(d, 0);
        return mu;
    };

    const int scale = 6, cell = 28 * scale, gap = 6;

    // ---- Demo 1: reconstructions (deterministic, using mu) ----
    {
        const int n = 8;
        Bitmap img(n * cell + (n - 1) * gap, 2 * cell + gap);
        for (int i = 0; i < n; ++i) {
            std::size_t col = static_cast<std::size_t>(i);
            Matrix<float> x  = X_test.selectColumns(std::vector<std::size_t>{col});
            Matrix<float> mu = encode_mu(X_test, col);
            Matrix<float> r  = dec.forward(mu);
            const int x0 = i * (cell + gap);
            blit_digit(img, x, 0, x0, 0,          scale);
            blit_digit(img, r, 0, x0, cell + gap, scale);
        }
        save_bmp("vae_reconstructions.bmp", img);
        std::cout << "\nwrote vae_reconstructions.bmp ("
                  << img.width() << "x" << img.height()
                  << ", top = originals, bottom = reconstructions)\n";
    }

    // ---- Demo 2: latent grid -- only meaningful for a 2D latent ----
    if constexpr (LATENT == 2) {
        const int   n = 18;          // n x n grid of digits
        const float r = 2.5f;        // sweep each axis over [-r, r]
        const int   s = 2;           // small per-cell scale; no gaps -> continuous sheet
        Bitmap img(n * 28 * s, n * 28 * s);
        for (int i = 0; i < n; ++i) {           // rows: z2 from +r (top) to -r
            for (int j = 0; j < n; ++j) {       // cols: z1 from -r to +r
                Matrix<float> z(LATENT, 1);
                z(0, 0) = -r + 2 * r * j / (n - 1);
                z(1, 0) =  r - 2 * r * i / (n - 1);
                Matrix<float> d = dec.forward(z);
                blit_digit(img, d, 0, j * 28 * s, i * 28 * s, s);
            }
        }
        save_bmp("vae_latent_grid.bmp", img);
        std::cout << "wrote vae_latent_grid.bmp (" << img.width() << "x" << img.height()
                  << ", the digit manifold over z in [-" << r << ", " << r << "]^2)\n";
    } else {
        std::cout << "skipped latent grid (drawn only for LATENT==2; current LATENT="
                  << LATENT << ")\n";
    }

    // ---- Demo 2b: latent interpolation (works in any dimension) ----
    {
        const int steps = 8;
        Matrix<float> za = encode_mu(X_test, find_digit(y_test, 3));
        Matrix<float> zb = encode_mu(X_test, find_digit(y_test, 8));
        Bitmap img(steps * cell + (steps - 1) * gap, cell);
        for (int s = 0; s < steps; ++s) {
            const float t = static_cast<float>(s) / (steps - 1);
            Matrix<float> z = za * (1.0f - t) + zb * t;
            Matrix<float> r = dec.forward(z);
            blit_digit(img, r, 0, s * (cell + gap), 0, scale);
        }
        save_bmp("vae_interpolation.bmp", img);
        std::cout << "wrote vae_interpolation.bmp (" << img.width() << "x" << img.height()
                  << ", latent morph 3 -> 8 through mu)\n";
    }

    // ---- Demo 3: random samples z ~ N(0, I) ----
    {
        const int n = 10;
        Bitmap img(n * cell + (n - 1) * gap, cell);
        Matrix<float> Z(LATENT, n);
        Z.randomizeNormal(0.0f, 1.0f, /*seed=*/7);
        for (int i = 0; i < n; ++i) {
            Matrix<float> z(LATENT, 1);
            for (std::size_t d = 0; d < LATENT; ++d) z(d, 0) = Z(d, i);
            Matrix<float> d = dec.forward(z);
            blit_digit(img, d, 0, i * (cell + gap), 0, scale);
        }
        save_bmp("vae_samples.bmp", img);
        std::cout << "wrote vae_samples.bmp (" << img.width() << "x" << img.height()
                  << ", digits decoded from random z ~ N(0, I))\n";
    }

    return 0;
}
