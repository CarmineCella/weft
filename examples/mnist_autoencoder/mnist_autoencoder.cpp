// mnist_autoencoder / mnist_autoencoder.cpp
//
// An autoencoder on MNIST.  Unsupervised: there are no labels in the
// training objective -- the network is trained to reproduce its own
// input through a narrow bottleneck, forcing it to learn a compact
// representation of "what a digit looks like".
//
// The model is two Networks:
//
//   encoder:  784 -> 128 -> ReLU -> 32 -> ReLU -> BOTTLENECK   (linear)
//   decoder:  BOTTLENECK -> 32 -> ReLU -> 128 -> ReLU -> 784 -> Sigmoid
//
// Splitting into two Networks (rather than one deep stack) lets us:
//   - encode an input to its latent code
//   - decode a latent code back to an image
//   - interpolate between two codes and watch one digit morph into
//     another, which is the payoff demo.
//
// Loss is MSE between the reconstruction and the original input.
// Labels are used ONLY to pick specific digits for the morph demo.
//
// Output is written as BMP images (via Bmp.h) into the current
// directory:
//   ae_reconstructions.bmp  -- originals (top row) vs reconstructions
//   ae_interpolation.bmp    -- a digit morph through latent space
//
// Usage:  mnist_autoencoder [data_dir]   (default ../data/mnist)
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
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace weft;

constexpr std::size_t BOTTLENECK = 16;

// Draw a 28x28 image (column `col` of a 784xN matrix, values in [0,1])
// into `bmp` with its top-left corner at (x0, y0), scaled up by an
// integer factor (nearest-neighbour) so it's comfortably visible.
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
    const std::string data_dir = (argc > 1) ? argv[1] : "../../../data/mnist";

    std::cout << "weft :: MNIST autoencoder\n";
    std::cout << "data dir: " << data_dir << "\n\n";

    // ---- Load MNIST (labels only used for the morph demo) ----
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

    // ---- Build encoder and decoder ----
    Network<float> enc;
    enc.add<Dense>(784, 128);
    enc.add<ReLU>();
    enc.add<Dense>(128, 32);
    enc.add<ReLU>();
    enc.add<Dense>(32, BOTTLENECK);     // linear bottleneck

    Network<float> dec;
    dec.add<Dense>(BOTTLENECK, 32);
    dec.add<ReLU>();
    dec.add<Dense>(32, 128);
    dec.add<ReLU>();
    dec.add<Dense>(128, 784);
    dec.add<Sigmoid>();                 // outputs in (0, 1) to match pixels

    MSE<float>  mse;
    Adam<float> opt(1e-3f);             // one optimiser tracks both nets' params

    const std::size_t batch_size = 128;
    const int         epochs     = 20;

    std::cout << "architecture:\n"
              << "    encoder: 784 -> 128 -> ReLU -> 32 -> ReLU -> " << BOTTLENECK << "\n"
              << "    decoder: " << BOTTLENECK << " -> 32 -> ReLU -> 128 -> ReLU -> 784 -> Sigmoid\n"
              << "loss:        MSE (target = input)\n"
              << "optimiser:   Adam (lr=1e-3)\n"
              << "batch:       " << batch_size << "\n"
              << "epochs:      " << epochs << "\n\n";

    // ---- Training loop (unsupervised: target is the input itself) ----
    std::cout << "epoch  train MSE   test MSE   time\n";
    std::cout << "-----  ---------   --------   -----\n";

    auto t_start = clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_epoch = clock::now();
        enc.train(); dec.train();
        auto idx = shuffled_indices(X_train.cols(), /*seed=*/epoch);

        double epoch_loss = 0;
        std::size_t n_seen = 0;

        for (std::size_t start = 0; start < X_train.cols(); start += batch_size) {
            const std::size_t end = std::min(start + batch_size, X_train.cols());
            std::vector<std::size_t> bidx(idx.begin() + start, idx.begin() + end);
            const std::size_t bs = bidx.size();

            Matrix<float> Xb = X_train.selectColumns(bidx);

            Matrix<float> z     = enc.forward(Xb);
            Matrix<float> recon = dec.forward(z);
            const float   L     = mse.forward(recon, Xb);   // target = input

            Matrix<float> dRecon = mse.backward();
            Matrix<float> dz     = dec.backward(dRecon);
            enc.backward(dz);

            dec.update(opt);
            enc.update(opt);

            epoch_loss += static_cast<double>(L) * bs;
            n_seen     += bs;
        }
        const float train_mse = static_cast<float>(epoch_loss / n_seen);

        // Test MSE (eval mode; no dropout here but good practice).
        enc.eval(); dec.eval();
        double test_loss = 0;
        std::size_t test_seen = 0;
        for (std::size_t start = 0; start < X_test.cols(); start += batch_size) {
            const std::size_t end = std::min(start + batch_size, X_test.cols());
            std::vector<std::size_t> bidx(end - start);
            for (std::size_t i = 0; i < bidx.size(); ++i) bidx[i] = start + i;
            Matrix<float> Xb = X_test.selectColumns(bidx);
            Matrix<float> recon = dec.forward(enc.forward(Xb));
            test_loss += static_cast<double>(mse.forward(recon, Xb)) * bidx.size();
            test_seen += bidx.size();
        }
        const float test_mse = static_cast<float>(test_loss / test_seen);

        double secs = std::chrono::duration<double>(clock::now() - t_epoch).count();
        std::cout << std::fixed << std::setprecision(6)
                  << std::setw(5)  << epoch     << "  "
                  << std::setw(9)  << train_mse << "   "
                  << std::setw(8)  << test_mse  << "   "
                  << std::setprecision(1) << std::setw(4) << secs << "s\n" << std::flush;
    }

    double total = std::chrono::duration<double>(clock::now() - t_start).count();
    std::cout << "\ntotal training time: " << std::setprecision(1) << total << "s\n";

    enc.eval(); dec.eval();

    const int scale = 6;
    const int cell  = 28 * scale;     // pixels per rendered digit
    const int gap   = 6;              // black gap between cells

    // ---- Demo 1: reconstructions (top row originals, bottom row recon) ----
    {
        const int n = 5;             // digits 0..4
        Bitmap img(n * cell + (n - 1) * gap, 2 * cell + gap);
        for (int d = 0; d < n; ++d) {
            std::size_t col = find_digit(y_test, d);
            Matrix<float> x = X_test.selectColumns(std::vector<std::size_t>{col});
            Matrix<float> r = dec.forward(enc.forward(x));
            const int x0 = d * (cell + gap);
            blit_digit(img, x, 0, x0, 0,          scale);   // original
            blit_digit(img, r, 0, x0, cell + gap, scale);   // reconstruction
        }
        save_bmp("ae_reconstructions.bmp", img);
        std::cout << "\nwrote ae_reconstructions.bmp ("
                  << img.width() << "x" << img.height()
                  << ", top row = originals, bottom row = reconstructions)\n";
    }

    // ---- Demo 2: latent interpolation (morph one digit into another) ----
    {
        const int steps = 8;
        std::size_t col_a = find_digit(y_test, 3);
        std::size_t col_b = find_digit(y_test, 8);
        Matrix<float> xa = X_test.selectColumns(std::vector<std::size_t>{col_a});
        Matrix<float> xb = X_test.selectColumns(std::vector<std::size_t>{col_b});

        Matrix<float> za = enc.forward(xa);   // (BOTTLENECK, 1)
        Matrix<float> zb = enc.forward(xb);

        Bitmap img(steps * cell + (steps - 1) * gap, cell);
        for (int s = 0; s < steps; ++s) {
            const float t = static_cast<float>(s) / (steps - 1);
            Matrix<float> z = za * (1.0f - t) + zb * t;     // straight line in latent space
            Matrix<float> r = dec.forward(z);
            blit_digit(img, r, 0, s * (cell + gap), 0, scale);
        }
        save_bmp("ae_interpolation.bmp", img);
        std::cout << "wrote ae_interpolation.bmp ("
                  << img.width() << "x" << img.height()
                  << ", left = 3, right = 8, middle = latent-space blends)\n";
    }

    return 0;
}
