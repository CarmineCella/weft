// sound_classifier / sound_classifier.cpp
//
// Train an MLP on a feature cache produced by extract_features.
//
//   1. extract_features <wav_dir> sol_mfcc.feat   mfcc      # slow, once
//   2. sound_classifier sol_mfcc.feat                       # fast, iterate
//
// The classifier picks its architecture from the feature_type stored in
// the cache, so the two CLIs stay in sync.
//
#include "Adam.h"
#include "CrossEntropy.h"
#include "Data.h"
#include "Dense.h"
#include "Dropout.h"
#include "FeatureCache.h"
#include "Network.h"
#include "ReLU.h"
#include "Softmax.h"
#include "Standardizer.h"

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

static std::pair<float, float>
evaluate(Network<float>& net, CrossEntropy<float>& ce,
         const Matrix<float>& X, const Matrix<float>& Y,
         std::size_t batch_size)
{
    const std::size_t N = X.cols();
    double      sum_loss = 0;
    std::size_t correct  = 0;
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
    return { static_cast<float>(sum_loss / N), static_cast<float>(correct) / N };
}

int main(int argc, char** argv) {
    using clock = std::chrono::steady_clock;

    if (argc < 2) {
        std::cerr << "usage: sound_classifier <feature_file>\n";
        return 1;
    }
    const std::string feature_path = argv[1];

    std::cout << "weft :: sound classifier\n";
    std::cout << "feature file: " << feature_path << "\n";

    // ---- Load cache ----
    CachedFeatures cache;
    try {
        cache = load_features(feature_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const std::size_t FEAT_DIM = cache.X.rows();
    const std::size_t N        = cache.X.cols();
    const int         K        = static_cast<int>(cache.class_names.size());

    std::cout << "feature type: " << cache.feature_type
              << " (" << FEAT_DIM << " dim)\n";
    std::cout << "samples:      " << N << "\n";
    std::cout << "classes:      " << K << "\n\n";

    // ---- Class breakdown ----
    {
        std::cout << "discovered classes:\n";
        for (int id = 0; id < K; ++id) {
            std::size_t count = 0;
            for (int l : cache.labels) if (l == id) ++count;
            std::cout << "  " << std::setw(2) << id << "  "
                      << std::setw(8) << std::left << cache.class_names[id]
                      << std::right << "  " << count << " samples\n";
        }
        std::cout << "\n";
    }

    // ---- Train/test split + standardise ----
    Matrix<float> Y = one_hot<float>(cache.labels, K);

    auto split = train_test_split(cache.X, Y, 0.2f, /*seed=*/1);
    std::cout << "split: " << split.X_train.cols() << " train, "
              << split.X_test.cols()  << " test  (seed=1)\n";

    Standardizer<float> scaler;
    Matrix<float> X_tr = scaler.fit_transform(split.X_train);
    Matrix<float> X_te = scaler.transform   (split.X_test);
    std::cout << "features standardised (fit on train only)\n\n";

    // ---- Architecture: size-appropriate to the feature type ----
    Network<float> net;
    if (cache.feature_type == "logmag") {
        net.add<Dense>  (FEAT_DIM, 256);
        net.add<ReLU>   ();
        net.add<Dropout>(0.3f, /*seed=*/1);
        net.add<Dense>  (256, 128);
        net.add<ReLU>   ();
        net.add<Dropout>(0.3f, /*seed=*/2);
        net.add<Dense>  (128, K);
        net.add<Softmax>();
        std::cout << "architecture:\n"
                  << "    Dense(" << FEAT_DIM << ", 256) -> ReLU -> Dropout(0.3)\n"
                  << "    Dense(256, 128)  -> ReLU -> Dropout(0.3)\n"
                  << "    Dense(128, " << K << ")    -> Softmax\n";
    } else if (cache.feature_type == "mfcc") {
        net.add<Dense>  (FEAT_DIM, 64);
        net.add<ReLU>   ();
        net.add<Dropout>(0.2f, /*seed=*/1);
        net.add<Dense>  (64, K);
        net.add<Softmax>();
        std::cout << "architecture:\n"
                  << "    Dense(" << FEAT_DIM << ", 64)  -> ReLU -> Dropout(0.2)\n"
                  << "    Dense(64, " << K << ")    -> Softmax\n";
    } else {
        std::cerr << "unknown feature type in cache: " << cache.feature_type << "\n";
        return 1;
    }

    CrossEntropy<float> ce;
    Adam<float>         opt(1e-3f);

    const std::size_t batch_size = 32;
    const int         epochs     = 100;

    std::cout << "loss:       cross-entropy\n"
              << "optimiser:  Adam (lr=1e-3)\n"
              << "batch:      " << batch_size << "\n"
              << "epochs:     " << epochs << "\n\n";

    // ---- Training loop ----
    std::cout << "epoch  train loss  train acc   test loss   test acc    time\n";
    std::cout << "-----  ----------  ---------   ---------   --------   -----\n";

    auto t_start = clock::now();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t_epoch = clock::now();

        net.train();
        auto idx = shuffled_indices(X_tr.cols(), /*seed=*/epoch);

        double      epoch_loss    = 0;
        std::size_t epoch_correct = 0;
        std::size_t n_seen        = 0;

        for (std::size_t start = 0; start < X_tr.cols(); start += batch_size) {
            const std::size_t end = std::min(start + batch_size, X_tr.cols());
            std::vector<std::size_t> batch_idx(idx.begin() + start, idx.begin() + end);
            const std::size_t bs = batch_idx.size();

            Matrix<float> Xb = X_tr.selectColumns(batch_idx);
            Matrix<float> Yb = split.Y_train.selectColumns(batch_idx);

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

        net.eval();
        auto [test_loss, test_acc] = evaluate(net, ce, X_te, split.Y_test, batch_size);

        double secs = std::chrono::duration<double>(clock::now() - t_epoch).count();

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(5)  << epoch        << "  "
                  << std::setw(10) << train_loss   << "  "
                  << std::setw(9)  << train_acc    << "   "
                  << std::setw(9)  << test_loss    << "   "
                  << std::setw(8)  << test_acc     << "   "
                  << std::setprecision(1) << std::setw(4) << secs << "s"
                  << "\n" << std::flush;
    }

    double total_secs = std::chrono::duration<double>(clock::now() - t_start).count();

    net.eval();
    auto [final_loss, final_acc] = evaluate(net, ce, X_te, split.Y_test, batch_size);

    std::cout << "\ntotal training time: " << std::setprecision(1)
              << total_secs << "s\n"
              << std::setprecision(4)
              << "final test accuracy: " << final_acc
              << "  (" << static_cast<int>(final_acc * X_te.cols() + 0.5f)
              << " / " << X_te.cols() << " correct)\n";

    return 0;
}
