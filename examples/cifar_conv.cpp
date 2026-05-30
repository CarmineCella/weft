// cifar_conv.cpp
//
// CIFAR-10 conv classifier.  The first real test that all the conv work
// from notes 18-19 actually trains on natural images.
//
// Architecture (AlexNet-shape, just much smaller):
//
//   input  (N, 3, 32, 32)
//     Conv2D(3 -> 32, k=5, p=2)     -> (N, 32, 32, 32)
//     ReLU4D
//     MaxPool2D(2)                  -> (N, 32, 16, 16)
//     Conv2D(32 -> 64, k=3, p=1)    -> (N, 64, 16, 16)
//     ReLU4D
//     MaxPool2D(2)                  -> (N, 64, 8, 8)
//   ---- flatten ----                -> (4096, N)
//     Dense(4096, 256)              -- hidden layer absorbs dense params
//     ReLU
//     Dropout(0.5)                  -- regularises the dense head
//     Dense(256, 10)
//     Softmax
//
// The dense head used to be a single Dense(4096, 10) which overfit
// badly (99% train, 60% test).  Adding a 256-unit bottleneck with
// dropout helped, but with only ~1M dense parameters versus 16-50K
// training images, the model still overfits.  The further fix is data
// augmentation: each training batch is randomly horizontally flipped
// and randomly translated by up to 4 pixels (zero-filled).  These two
// together are the standard CIFAR-10 augmentation, and they push test
// accuracy from ~65% to ~75-78% on this small architecture.
//
// IMPORTANT: run without subsample for best results.  Using
// `subsample 3` keeps only 16667 of the 50000 training images, which
// hurts test accuracy significantly even with dropout + augmentation.
//
// The first conv layer's 5x5 kernels are wide enough to show
// recognisable orientation and colour patterns once trained -- that's
// what the companion example cifar_filters.cpp visualises.
//
// Usage:
//   cifar_conv <cifar-10-batches-bin/> [epochs] [model_prefix] [subsample]
//
// Saves <prefix>.conv and <prefix>.dense after every epoch; Ctrl-C
// during training also saves before exiting.

#include "Tensor4D.h"
#include "ConvNetwork.h"
#include "Conv2D.h"
#include "ReLU4D.h"
#include "MaxPool2D.h"
#include "Flatten.h"
#include "Network.h"
#include "Dense.h"
#include "ReLU.h"
#include "Dropout.h"
#include "Softmax.h"
#include "CrossEntropy.h"
#include "Adam.h"
#include "CIFAR10.h"
#include "Data.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace weft;

constexpr std::size_t N_CLASSES = 10;

// ---------------------------------------------------------------------
// Memcpy-based batch construction.  Each example in the source Tensor4D
// occupies C*H*W contiguous floats in the flat buffer (NCHW), so we can
// just copy one image at a time rather than the four-nested-loop dance.
// ---------------------------------------------------------------------
static Tensor4D<float> select_batch(const Tensor4D<float>& src,
                                    const std::vector<std::size_t>& idx) {
    const std::size_t img_size = src.C() * src.H() * src.W();
    Tensor4D<float> out(idx.size(), src.C(), src.H(), src.W());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        std::memcpy(out.data() + i * img_size,
                    src.data() + idx[i] * img_size,
                    img_size * sizeof(float));
    }
    return out;
}

// ---------------------------------------------------------------------
// Data augmentation.  Applied to each training batch (not to the test
// set).  Two standard CIFAR-10 tricks:
//
//   1. random horizontal flip:  each image independently flipped left-
//      to-right with probability 0.5.
//
//   2. random crop with zero-padding:  conceptually pad each image to
//      (H+2p, W+2p) with zeros, then crop back to (H, W) from a random
//      position.  Implemented as a single-pass per-channel shift that
//      doesn't materialize the padded tensor.
//
// Together these typically add 5-10 percentage points of test accuracy
// on small CIFAR-10 nets, by giving the network many slightly-different
// versions of each training image and so making it harder to memorise.
// ---------------------------------------------------------------------
static void random_h_flip(Tensor4D<float>& batch, std::mt19937& rng) {
    std::bernoulli_distribution flip(0.5);
    const std::size_t H = batch.H();
    const std::size_t W = batch.W();
    for (std::size_t n = 0; n < batch.N(); ++n) {
        if (!flip(rng)) continue;
        for (std::size_t c = 0; c < batch.C(); ++c) {
            for (std::size_t h = 0; h < H; ++h) {
                for (std::size_t w = 0; w < W / 2; ++w) {
                    std::swap(batch(n, c, h, w),
                              batch(n, c, h, W - 1 - w));
                }
            }
        }
    }
}

static void random_crop(Tensor4D<float>& batch, std::mt19937& rng, int pad = 4) {
    const int H = static_cast<int>(batch.H());
    const int W = static_cast<int>(batch.W());
    std::uniform_int_distribution<int> off(-pad, pad);

    Tensor4D<float> out(batch.N(), batch.C(), H, W);
    for (std::size_t n = 0; n < batch.N(); ++n) {
        const int dy = off(rng);   // shift in rows
        const int dx = off(rng);   // shift in cols
        for (std::size_t c = 0; c < batch.C(); ++c) {
            for (int h = 0; h < H; ++h) {
                const int sh = h + dy;
                for (int w = 0; w < W; ++w) {
                    const int sw = w + dx;
                    out(n, c, h, w) =
                        (sh >= 0 && sh < H && sw >= 0 && sw < W)
                            ? batch(n, c, sh, sw)
                            : 0.0f;
                }
            }
        }
    }
    batch = std::move(out);
}

// Returns predicted class for one column of a probability matrix.
static std::size_t argmax_column(const Matrix<float>& P, std::size_t col) {
    std::size_t best = 0;
    float       v    = P(0, col);
    for (std::size_t c = 1; c < P.rows(); ++c)
        if (P(c, col) > v) { v = P(c, col); best = c; }
    return best;
}

// ---------------------------------------------------------------------
// Test-set accuracy.  Pure forward pass, batched to keep memory bounded.
// Switches both networks to eval() mode (turns off dropout) and back to
// train() on exit -- this matters now that the dense head has Dropout.
// ---------------------------------------------------------------------
static float test_accuracy(ConvNetwork<float>& conv, Network<float>& dense,
                           const Tensor4D<float>& X, const std::vector<int>& y,
                           std::size_t batch_size) {
    conv .eval();
    dense.eval();

    const std::size_t N = X.N();
    std::size_t correct = 0;

    for (std::size_t bs = 0; bs < N; bs += batch_size) {
        const std::size_t be = std::min(N, bs + batch_size);
        std::vector<std::size_t> idx(be - bs);
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = bs + i;

        Tensor4D<float> Xb    = select_batch(X, idx);
        Tensor4D<float> feat  = conv.forward(Xb);
        Matrix<float>   flat  = flatten(feat);
        Matrix<float>   probs = dense.forward(flat);

        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (static_cast<int>(argmax_column(probs, i)) == y[idx[i]]) ++correct;
        }
    }

    conv .train();
    dense.train();
    return 100.0f * static_cast<float>(correct) / static_cast<float>(N);
}

// Ctrl-C: set a flag the training loop checks each batch so we can save
// a partial checkpoint before exit.
static volatile std::sig_atomic_t g_interrupted = 0;
static void on_sigint(int) { g_interrupted = 1; }


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr <<
          "usage: cifar_conv <cifar-10-batches-bin/> [epochs] [prefix] [subsample]\n"
          "  data_dir   path to the CIFAR-10 binary directory (use\n"
          "             data/download_cifar10.sh to fetch it)\n"
          "  epochs     number of training epochs (default 15)\n"
          "  prefix     output model file prefix  (default 'cifar_model')\n"
          "  subsample  use every Nth training example (default 1).  E.g.\n"
          "             '10' uses 5000 train images instead of 50000, for\n"
          "             quick smoke tests.\n";
        return 1;
    }

    const std::string data_dir  = argv[1];
    const int         EPOCHS    = (argc > 2) ? std::atoi(argv[2]) : 15;
    const std::string prefix    = (argc > 3) ? argv[3] : "cifar_model";
    const int         subsample = (argc > 4) ? std::atoi(argv[4]) : 1;

    std::signal(SIGINT, on_sigint);

    std::cout << "weft :: cifar_conv\n"
              << "  data:      " << data_dir  << "\n"
              << "  epochs:    " << EPOCHS    << "\n"
              << "  prefix:    " << prefix    << "\n"
              << "  subsample: " << subsample << "\n\n";

    // ---- Load data --------------------------------------------------
    std::cout << "Loading CIFAR-10...\n";
    auto train = load_cifar10<float>(cifar10_train_paths(data_dir));
    auto test  = load_cifar10<float>(cifar10_test_paths (data_dir));
    std::cout << "  train: " << train.images.N() << " images\n"
              << "  test:  " << test.images.N()  << " images\n";

    if (subsample > 1) {
        std::vector<std::size_t> keep;
        for (std::size_t i = 0; i < train.images.N(); i += subsample) keep.push_back(i);
        Tensor4D<float> X_sub = select_batch(train.images, keep);
        std::vector<int> y_sub; y_sub.reserve(keep.size());
        for (std::size_t i : keep) y_sub.push_back(train.labels[i]);
        train.images = std::move(X_sub);
        train.labels = std::move(y_sub);
        std::cout << "  after subsample 1/" << subsample
                  << ": " << train.images.N() << " train images\n";
    }
    std::cout << "\n";

    Matrix<float> Y_train = one_hot<float>(train.labels, N_CLASSES);

    // ---- Build network ----------------------------------------------
    ConvNetwork<float> conv;
    conv.add<Conv2D>  (/*in=*/3,  /*out=*/32, /*k=*/5, /*stride=*/1, /*pad=*/2);
    conv.add<ReLU4D>  ();
    conv.add<MaxPool2D>(2);
    conv.add<Conv2D>  (/*in=*/32, /*out=*/64, /*k=*/3, /*stride=*/1, /*pad=*/1);
    conv.add<ReLU4D>  ();
    conv.add<MaxPool2D>(2);

    Network<float> dense;
    dense.add<Dense>  (64 * 8 * 8, 256);
    dense.add<ReLU>   ();
    dense.add<Dropout>(0.5f);
    dense.add<Dense>  (256, N_CLASSES);
    dense.add<Softmax>();

    std::cout << "Architecture:\n";
    std::cout << conv.summary();
    std::cout << "  -- flatten -- (Tensor4D -> Matrix)\n";
    std::cout << dense.summary();
    std::cout << "\n";

    CrossEntropy<float> loss;
    Adam<float>         opt;

    const std::size_t BATCH_SIZE = 64;
    const std::size_t N          = train.images.N();

    auto save_model = [&]() {
        conv .save(prefix + ".conv");
        dense.save(prefix + ".dense");
    };

    std::cout << "Starting training...\n";
    const auto t0 = std::chrono::steady_clock::now();

    // Augmentation RNG.  Seeded fixed so reruns are reproducible; the
    // shuffle uses its own seed (the epoch number) above.
    std::mt19937 aug_rng(12345);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto perm = shuffled_indices(N, static_cast<unsigned>(epoch));

        float       loss_sum = 0.0f;
        std::size_t correct  = 0;
        std::size_t seen     = 0;
        std::size_t b_idx    = 0;

        for (std::size_t bs = 0; bs < N; bs += BATCH_SIZE, ++b_idx) {
            const std::size_t be = std::min(N, bs + BATCH_SIZE);
            std::vector<std::size_t> batch(perm.begin() + bs, perm.begin() + be);

            // ---- forward ----
            Tensor4D<float> Xb    = select_batch(train.images, batch);

            // ---- augment (training only) ----
            // Each image gets a 50% chance of horizontal flip, plus a
            // small random translation with zero-fill at the edges.
            // No-ops for the test pass (which never calls these).
            random_h_flip(Xb, aug_rng);
            random_crop (Xb, aug_rng, /*pad=*/4);
            Matrix<float>   Yb    = Y_train.selectColumns(batch);

            Tensor4D<float> feat  = conv.forward(Xb);
            Matrix<float>   flat  = flatten(feat);
            Matrix<float>   probs = dense.forward(flat);

            const float L = loss.forward(probs, Yb);
            loss_sum += L * batch.size();
            seen     += batch.size();

            for (std::size_t i = 0; i < batch.size(); ++i)
                if (static_cast<int>(argmax_column(probs, i)) == train.labels[batch[i]])
                    ++correct;

            // ---- backward ----
            Matrix<float>   dProbs = loss.backward();
            Matrix<float>   dFlat  = dense.backward(dProbs);
            Tensor4D<float> dFeat  = unflatten(dFlat,
                                               feat.N(), feat.C(),
                                               feat.H(), feat.W());
            conv.backward(dFeat);

            conv .update(opt);
            dense.update(opt);

            if (g_interrupted) {
                std::cout << "\n  interrupted -- saving and exiting\n";
                save_model();
                return 0;
            }
        }

        const auto   t_now    = std::chrono::steady_clock::now();
        const double elapsed  = std::chrono::duration<double>(t_now - t0).count();
        const float  train_acc= 100.0f * static_cast<float>(correct) / static_cast<float>(seen);
        const float  avg_loss = loss_sum / static_cast<float>(seen);
        const float  test_acc = test_accuracy(conv, dense, test.images, test.labels, BATCH_SIZE);

        std::cout << "epoch " << std::setw(2) << epoch
                  << "  loss="      << std::fixed << std::setprecision(4) << avg_loss
                  << "  train_acc=" << std::setprecision(2) << train_acc << "%"
                  << "  test_acc="  << test_acc  << "%"
                  << "  elapsed="   << std::setprecision(1) << elapsed << "s"
                  << "\n";

        save_model();
    }

    std::cout << "\nDone. Model saved as "
              << prefix << ".conv / " << prefix << ".dense\n"
              << "Run cifar_filters to visualise the first-layer kernels.\n";
    return 0;
}
