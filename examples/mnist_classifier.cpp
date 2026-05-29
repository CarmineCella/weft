// mnist_classifier / mnist_classifier.cpp
//
// Train a 3-layer MLP on MNIST with Adam, ReLU, and Dropout.  This is
// the first example in weft that puts every piece of the library to work
// at once:
//
//   - MNIST.h loader for the IDX file format
//   - Network composed of Dense + ReLU + Dropout + Softmax
//   - Cross-entropy loss
//   - Adam optimiser (per-parameter adaptive learning rates)
//   - train() / eval() mode switching so Dropout is only active on the
//     training pass, never on the test pass.
//
// Architecture:
//     Dense(784, 256) -> ReLU -> Dropout(0.3)
//     Dense(256, 128) -> ReLU -> Dropout(0.3)
//     Dense(128, 10)  -> Softmax
//
// Usage:
//     mnist_classifier [data_dir]
//
// data_dir defaults to ../data/mnist (relative to build/, the standard
// place Make invokes binaries from).  The directory must contain the
// four uncompressed IDX files; use data/download_mnist.sh to fetch them.
//
#include "Adam.h"
#include "CrossEntropy.h"
#include "Data.h"
#include "Dense.h"
#include "Dropout.h"
#include "MNIST.h"
#include "Network.h"
#include "ReLU.h"
#include "Softmax.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace weft;

// Evaluate (loss, accuracy) over X, Y in mini-batches so we never
// allocate a giant 10000-column intermediate.  net should already be
// in eval mode when this is called -- the function does not touch it.
static std::pair<float, float>
evaluate(Network<float>& net,
         CrossEntropy<float>& ce,
         const Matrix<float>& X,
         const Matrix<float>& Y,
         std::size_t batch_size)
{
    const std::size_t N = X.cols();
    double sum_loss = 0;     // double for accumulation, even though we work in float
    std::size_t correct = 0;

    for (std::size_t start = 0; start < N; start += batch_size) {
        const std::size_t end = std::min(start + batch_size, N);
        const std::size_t bs  = end - start;

        std::vector<std::size_t> idx(bs);
        for (std::size_t i = 0; i < bs; ++i) idx[i] = start + i;

        Matrix<float> Xb = X.selectColumns(idx);
        Matrix<float> Yb = Y.selectColumns(idx);
        Matrix<float> S  = net.forward(Xb);

        sum_loss += static_cast<double>(ce.forward(S, Yb)) * bs;
        correct  += static_cast<std::size_t>(accuracy<float>(S, Yb) * bs + 0.5f);
    }
    return { static_cast<float>(sum_loss / N),
             static_cast<float>(correct) / N };
}

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;

    const std::string data_dir = (argc > 1) ? argv[1] : "../data/mnist";

    std::cout << "weft :: MNIST classifier\n";
    std::cout << "data directory: " << data_dir << "\n\n";

    // ----------------------------------------------------------------
    // 1. Load MNIST.  Each load_images call gives a (784, N) matrix
    //    with pixels normalised to [0, 1].
    // ----------------------------------------------------------------
    Matrix<float>    X_train, X_test;
    std::vector<int> y_train, y_test;
    try {
        X_train = mnist::load_images<float>(data_dir + "/train-images-idx3-ubyte");
        y_train = mnist::load_labels       (data_dir + "/train-labels-idx1-ubyte");
        X_test  = mnist::load_images<float>(data_dir + "/t10k-images-idx3-ubyte");
        y_test  = mnist::load_labels       (data_dir + "/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "\nerror loading MNIST: " << e.what() << "\n";
        std::cerr << "did you run data/download_mnist.sh ?\n";
        return 1;
    }

    Matrix<float> Y_train = one_hot<float>(y_train, 10);
    Matrix<float> Y_test  = one_hot<float>(y_test,  10);

    std::cout << "loaded "
              << X_train.cols() << " training and "
              << X_test.cols()  << " test images "
              << "(" << X_train.rows() << " pixels each)\n\n";

    // ----------------------------------------------------------------
    // 2. Build the network.
    //    Pixels are already in [0, 1], so no Standardizer needed; He
    //    init plus ReLU is happy with that range.
    // ----------------------------------------------------------------
    Network<float> net;
    net.add<Dense>  (784, 256);
    net.add<ReLU>   ();
    net.add<Dropout>(0.3f, /*seed=*/1);
    net.add<Dense>  (256, 128);
    net.add<ReLU>   ();
    net.add<Dropout>(0.3f, /*seed=*/2);
    net.add<Dense>  (128, 10);
    net.add<Softmax>();

    CrossEntropy<float> ce;
    Adam<float>         opt(1e-3f);

    const std::size_t batch_size = 128;
    const int         epochs     = 10;

    std::cout << "architecture:\n" << net.summary() << "\n";
    std::cout << "loss:         cross-entropy\n";
    std::cout << "optimiser:    Adam (lr=1e-3)\n";
    std::cout << "batch size:   " << batch_size << "\n";
    std::cout << "epochs:       " << epochs     << "\n\n";

    // ----------------------------------------------------------------
    // 3. Baseline:  before any training, the network is at chance (~10%
    //    on a 10-class problem).  Showing this anchors the per-epoch
    //    numbers below.
    // ----------------------------------------------------------------
    net.eval();
    {
        auto [init_loss, init_acc] = evaluate(net, ce, X_test, Y_test, batch_size);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "before training:   test loss = " << init_loss
                  << ",   test acc = " << init_acc << "\n\n";
    }

    // ----------------------------------------------------------------
    // 4. Training loop.
    //    Train metrics are accumulated DURING the training pass, so
    //    they reflect dropout being active (which slightly hurts the
    //    metric, especially early in training).  Test metrics are
    //    measured in eval mode after each epoch.
    // ----------------------------------------------------------------
    std::cout << "epoch  train loss  train acc   test loss   test acc    time\n";
    std::cout << "-----  ----------  ---------   ---------   --------   -----\n";

    auto t_total_start = clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_epoch_start = clock::now();

        // ---- training pass ----
        net.train();
        auto idx = shuffled_indices(X_train.cols(), /*seed=*/epoch);

        double      epoch_loss    = 0;
        std::size_t epoch_correct = 0;
        std::size_t n_seen        = 0;

        for (std::size_t start = 0; start < X_train.cols(); start += batch_size) {
            const std::size_t end = std::min(start + batch_size, X_train.cols());
            std::vector<std::size_t> batch_idx(idx.begin() + start, idx.begin() + end);
            const std::size_t bs = batch_idx.size();

            Matrix<float> Xb = X_train.selectColumns(batch_idx);
            Matrix<float> Yb = Y_train.selectColumns(batch_idx);

            Matrix<float> S = net.forward(Xb);
            const float   L = ce.forward(S, Yb);
            net.backward(ce.backward());
            net.update(opt);

            epoch_loss    += static_cast<double>(L) * bs;
            epoch_correct += static_cast<std::size_t>(accuracy<float>(S, Yb) * bs + 0.5f);
            n_seen        += bs;
        }

        const float train_loss = static_cast<float>(epoch_loss / n_seen);
        const float train_acc  = static_cast<float>(epoch_correct) / n_seen;

        // ---- test pass ----
        net.eval();
        auto [test_loss, test_acc] = evaluate(net, ce, X_test, Y_test, batch_size);

        auto t_epoch_end = clock::now();
        double secs = std::chrono::duration<double>(t_epoch_end - t_epoch_start).count();

        std::cout << std::setw(5)  << epoch        << "  "
                  << std::setw(10) << train_loss   << "  "
                  << std::setw(9)  << train_acc    << "   "
                  << std::setw(9)  << test_loss    << "   "
                  << std::setw(8)  << test_acc     << "   "
                  << std::setprecision(1) << std::setw(4) << secs << "s"
                  << std::setprecision(4)
                  << "\n" << std::flush;
    }

    auto t_total_end = clock::now();
    double total_secs = std::chrono::duration<double>(t_total_end - t_total_start).count();

    // ----------------------------------------------------------------
    // 5. Final report.
    // ----------------------------------------------------------------
    net.eval();
    auto [final_loss, final_acc] = evaluate(net, ce, X_test, Y_test, batch_size);

    std::cout << "\ntotal training time: " << std::setprecision(1) << total_secs << "s\n"
              << std::setprecision(4)
              << "final test accuracy: " << final_acc
              << "  (" << static_cast<int>(final_acc * X_test.cols() + 0.5f)
              << " / " << X_test.cols() << " correct)\n";

    return 0;
}
