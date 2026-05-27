// iris_classifier / main.cpp
//
// First end-to-end classifier in weft.  Trains a tiny MLP on Fisher's
// 1936 IRIS dataset and reports per-epoch loss and accuracy.
//
//   data:           150 samples, 4 features, 3 classes
//   architecture:   Dense(4,16) -> ReLU -> Dense(16,3) -> Softmax
//   loss:           cross-entropy
//   optimiser:      plain SGD with mini-batches
//
// The data flow demonstrates the standard ML pipeline:
//   1. load raw data into matrices
//   2. one-hot encode the labels
//   3. train/test split (random, with a fixed seed for reproducibility)
//   4. fit Standardizer on the TRAIN split only; apply to both splits
//      (fitting on test would leak test statistics into preprocessing)
//   5. mini-batch SGD with per-epoch shuffling
//   6. report loss and accuracy on both train and test each few epochs
//
#include "Matrix.h"
#include "Network.h"
#include "Dense.h"
#include "ReLU.h"
#include "Softmax.h"
#include "CrossEntropy.h"
#include "Standardizer.h"
#include "Data.h"
#include "SGD.h"

#include "iris_data.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace weft;

int main() {
    std::cout << "weft :: IRIS classifier\n\n";

    // ----------------------------------------------------------------
    // 1. Load the embedded dataset into a Matrix and a label vector.
    //    Convention: features as rows, examples as columns.
    // ----------------------------------------------------------------
    Matrix<float> X(iris::N_FEATURES, iris::N_SAMPLES);
    std::vector<int> labels(iris::N_SAMPLES);
    for (int j = 0; j < iris::N_SAMPLES; ++j) {
        for (int i = 0; i < iris::N_FEATURES; ++i)
            X(i, j) = iris::features[j][i];
        labels[j] = iris::labels[j];
    }
    Matrix<float> Y = one_hot<float>(labels, iris::N_CLASSES);

    std::cout << "loaded " << iris::N_SAMPLES << " examples, "
              << iris::N_FEATURES << " features, "
              << iris::N_CLASSES << " classes\n";

    // ----------------------------------------------------------------
    // 2. Train/test split. 20% test, fixed seed for reproducibility.
    //    Note: on a 30-example test set, accuracy is noisy w.r.t. the
    //    random split -- different seeds give anything from ~87% to 100%
    //    on this same model. Real evaluation would use k-fold CV.
    // ----------------------------------------------------------------
    auto split = train_test_split(X, Y, 0.2, /*seed=*/1);
    std::cout << "split: " << split.X_train.cols() << " train, "
              << split.X_test.cols()  << " test  (seed=1)\n";

    // ----------------------------------------------------------------
    // 3. Standardise. Fit on TRAIN only -- never on test.
    // ----------------------------------------------------------------
    Standardizer<float> scaler;
    Matrix<float> X_tr = scaler.fit_transform(split.X_train);
    Matrix<float> X_te = scaler.transform   (split.X_test);
    std::cout << "features standardised (mean ~ 0, std ~ 1 on train)\n\n";

    // ----------------------------------------------------------------
    // 4. Architecture and training hyperparameters.
    // ----------------------------------------------------------------
    Network<float> net;
    net.add<Dense>(iris::N_FEATURES, 16);
    net.add<ReLU>();
    net.add<Dense>(16, iris::N_CLASSES);
    net.add<Softmax>();
    CrossEntropy<float> ce;

    const int    epochs        = 100;
    const int    batch_size    = 16;
    const float  learning_rate = 0.10f;
    SGD<float>   opt(learning_rate);

    std::cout << "architecture: Dense(" << iris::N_FEATURES
              << ",16) -> ReLU -> Dense(16," << iris::N_CLASSES
              << ") -> Softmax\n";
    std::cout << "loss:         cross-entropy\n";
    std::cout << "optimiser:    plain SGD, lr=" << learning_rate
              << ", batch_size=" << batch_size << "\n";
    std::cout << "training for " << epochs << " epochs\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  epoch   train_loss   train_acc   test_loss   test_acc\n";
    std::cout << "  -----   ----------   ---------   ---------   --------\n";

    // ----------------------------------------------------------------
    // 5. Training loop.
    // ----------------------------------------------------------------
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // -- 5a. Train for one full epoch (skip on epoch 0 so the
        //        first reported row shows the untrained baseline).
        if (epoch > 0) {
            // Shuffle example indices for this epoch.
            auto idx = shuffled_indices(X_tr.cols(), /*seed=*/epoch);
            for (std::size_t start = 0; start < X_tr.cols(); start += batch_size) {
                const std::size_t end = std::min(start + batch_size, X_tr.cols());
                std::vector<std::size_t> batch_idx(idx.begin() + start,
                                                   idx.begin() + end);

                Matrix<float> X_batch = X_tr.selectColumns(batch_idx);
                Matrix<float> Y_batch = split.Y_train.selectColumns(batch_idx);

                // Standard 4-step training step:
                //   forward, loss, backward (loss -> net), update.
                Matrix<float> S = net.forward(X_batch);
                ce.forward(S, Y_batch);
                net.backward(ce.backward());
                net.update(opt);
            }
        }

        // -- 5b. Report every 10 epochs (and on the final epoch).
        if (epoch % 10 == 0 || epoch == epochs) {
            Matrix<float> S_tr = net.forward(X_tr);
            Matrix<float> S_te = net.forward(X_te);
            float L_tr = ce.forward(S_tr, split.Y_train);
            float L_te = ce.forward(S_te, split.Y_test);
            float A_tr = accuracy(S_tr, split.Y_train);
            float A_te = accuracy(S_te, split.Y_test);

            std::cout << "  " << std::setw(5) << epoch
                      << "   " << std::setw(10) << L_tr
                      << "   " << std::setw(9)  << A_tr
                      << "   " << std::setw(9)  << L_te
                      << "   " << std::setw(8)  << A_te << '\n';
        }
    }

    // ----------------------------------------------------------------
    // 6. Final summary.
    // ----------------------------------------------------------------
    Matrix<float> S_te = net.forward(X_te);
    float final_acc = accuracy(S_te, split.Y_test);

    std::cout << "\nfinal test accuracy: " << final_acc
              << "  (" << static_cast<int>(final_acc * split.X_test.cols())
              << "/" << split.X_test.cols() << " correct)\n";

    return 0;
}
