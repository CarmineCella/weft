// sound_classifier / sound_classifier.cpp
//
// Train an MLP on a feature cache produced by extract_features.
//
//   1. extract_features <wav_dir> sol_mfcc.feat   mfcc   averaged    # slow, once
//   2. sound_classifier sol_mfcc.feat                                # fast, iterate
//
// The classifier picks its architecture from the feature_type stored in
// the cache.  When the cache contains per-frame features, splitting is
// done by file_id (not by frame) so frames of the same file never appear
// in both train and test, and test predictions are aggregated per file
// at eval time.  A confusion matrix over the test set is printed at the
// end of training to show which class pairs get confused most.
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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace weft;

// Find argmax of a column of a matrix (or vector treated as one column).
static std::size_t argmax_col(const Matrix<float>& M, std::size_t j) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < M.rows(); ++i)
        if (M(i, j) > M(best, j)) best = i;
    return best;
}

// Aggregate per-frame softmax outputs into one mean vector per file.
// Returns:
//   averaged_preds: (K, n_files), each column = mean softmax over that file's frames
//   file_labels:    one int label per file (taken from the first frame of each file)
//   file_ids_out:   the unique file ids in the same order as averaged_preds columns
static void aggregate_per_file(const Matrix<float>& frame_preds,
                               const Matrix<float>& frame_targets,
                               const std::vector<int>& file_ids,
                               Matrix<float>& averaged_preds,
                               std::vector<int>& file_labels,
                               std::vector<int>& file_ids_out)
{
    const std::size_t K = frame_preds.rows();
    const std::size_t N = frame_preds.cols();

    // Walk frames in order; consecutive frames with the same file_id belong
    // to the same file (extract_features outputs them contiguously, and
    // group_train_test_split preserves order within X_test).
    std::vector<std::vector<std::size_t>> groups;  // groups[f] = list of frame indices
    std::map<int, std::size_t>            file_to_group;
    for (std::size_t j = 0; j < N; ++j) {
        auto it = file_to_group.find(file_ids[j]);
        if (it == file_to_group.end()) {
            file_to_group[file_ids[j]] = groups.size();
            groups.push_back({j});
            file_ids_out.push_back(file_ids[j]);
        } else {
            groups[it->second].push_back(j);
        }
    }

    averaged_preds = Matrix<float>(K, groups.size(), 0.0f);
    file_labels.resize(groups.size());

    for (std::size_t f = 0; f < groups.size(); ++f) {
        for (std::size_t j : groups[f])
            for (std::size_t k = 0; k < K; ++k)
                averaged_preds(k, f) += frame_preds(k, j);
        const float inv = 1.0f / groups[f].size();
        for (std::size_t k = 0; k < K; ++k)
            averaged_preds(k, f) *= inv;
        // Truth label of this file = argmax of first frame's one-hot column.
        file_labels[f] = static_cast<int>(argmax_col(frame_targets, groups[f][0]));
    }
}

// Evaluate (loss, accuracy) on the test set.  For per_frame mode, aggregate
// predictions by file so accuracy is reported per-file, not per-frame.
static std::pair<float, float>
evaluate(Network<float>& net, CrossEntropy<float>& ce,
         const Matrix<float>& X, const Matrix<float>& Y,
         const std::vector<int>& file_ids,
         bool per_frame,
         std::size_t batch_size)
{
    const std::size_t N = X.cols();

    // First pass: gather all frame-level predictions and frame-level loss.
    Matrix<float> all_preds(Y.rows(), N);
    double        sum_loss = 0;

    for (std::size_t start = 0; start < N; start += batch_size) {
        const std::size_t end = std::min(start + batch_size, N);
        const std::size_t bs  = end - start;
        std::vector<std::size_t> idx(bs);
        for (std::size_t i = 0; i < bs; ++i) idx[i] = start + i;
        Matrix<float> Xb = X.selectColumns(idx);
        Matrix<float> Yb = Y.selectColumns(idx);
        Matrix<float> S  = net.forward(Xb);
        sum_loss += static_cast<double>(ce.forward(S, Yb)) * bs;
        for (std::size_t i = 0; i < bs; ++i)
            for (std::size_t k = 0; k < S.rows(); ++k)
                all_preds(k, start + i) = S(k, i);
    }
    const float loss = static_cast<float>(sum_loss / N);

    if (!per_frame) {
        return { loss, accuracy<float>(all_preds, Y) };
    }

    // Per-file aggregation: average predictions across frames of each file.
    Matrix<float>    file_preds;
    std::vector<int> file_labels;
    std::vector<int> file_ids_out;
    aggregate_per_file(all_preds, Y, file_ids, file_preds, file_labels, file_ids_out);

    std::size_t correct = 0;
    for (std::size_t f = 0; f < file_preds.cols(); ++f) {
        std::size_t pred = argmax_col(file_preds, f);
        if (static_cast<int>(pred) == file_labels[f]) ++correct;
    }
    return { loss, static_cast<float>(correct) / file_preds.cols() };
}

// Build a (K x K) confusion matrix over the test set.  Rows = true class,
// columns = predicted class.  Returns:
//   confusion[true][pred] = count of test examples (frames if per_frame
//                          is false, files if per_frame is true)
static std::vector<std::vector<int>>
build_confusion(Network<float>& net,
                const Matrix<float>& X, const Matrix<float>& Y,
                const std::vector<int>& file_ids,
                bool per_frame, std::size_t K)
{
    std::vector<std::vector<int>> conf(K, std::vector<int>(K, 0));

    // Get all frame predictions.
    Matrix<float> preds(K, X.cols());
    const std::size_t batch_size = 128;
    for (std::size_t start = 0; start < X.cols(); start += batch_size) {
        const std::size_t end = std::min(start + batch_size, X.cols());
        const std::size_t bs  = end - start;
        std::vector<std::size_t> idx(bs);
        for (std::size_t i = 0; i < bs; ++i) idx[i] = start + i;
        Matrix<float> Xb = X.selectColumns(idx);
        Matrix<float> S  = net.forward(Xb);
        for (std::size_t i = 0; i < bs; ++i)
            for (std::size_t k = 0; k < K; ++k)
                preds(k, start + i) = S(k, i);
    }

    if (!per_frame) {
        for (std::size_t j = 0; j < preds.cols(); ++j) {
            const std::size_t pred  = argmax_col(preds, j);
            const std::size_t truth = argmax_col(Y,     j);
            conf[truth][pred]++;
        }
        return conf;
    }

    Matrix<float>    file_preds;
    std::vector<int> file_labels;
    std::vector<int> file_ids_out;
    aggregate_per_file(preds, Y, file_ids, file_preds, file_labels, file_ids_out);

    for (std::size_t f = 0; f < file_preds.cols(); ++f) {
        const std::size_t pred  = argmax_col(file_preds, f);
        const int         truth = file_labels[f];
        conf[truth][pred]++;
    }
    return conf;
}

// Pretty-print a confusion matrix with class names in headers.
static void print_confusion(const std::vector<std::vector<int>>& conf,
                            const std::vector<std::string>& class_names)
{
    const std::size_t K = conf.size();
    std::size_t name_w = 4;
    for (const auto& n : class_names) name_w = std::max(name_w, n.size());

    // Header row.
    std::cout << "\nconfusion matrix (rows=true, columns=predicted):\n";
    std::cout << "  " << std::string(name_w, ' ') << " |";
    for (std::size_t k = 0; k < K; ++k)
        std::cout << " " << std::setw(4) << class_names[k].substr(0, 4);
    std::cout << "\n  " << std::string(name_w, '-') << "-+";
    for (std::size_t k = 0; k < K; ++k) std::cout << "-----";
    std::cout << "\n";

    // Data rows.
    for (std::size_t t = 0; t < K; ++t) {
        std::cout << "  " << std::left << std::setw(name_w) << class_names[t]
                  << std::right << " |";
        for (std::size_t p = 0; p < K; ++p)
            std::cout << " " << std::setw(4) << conf[t][p];
        std::cout << "\n";
    }
    std::cout << "\n";
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
    std::cout << "mode:         " << (cache.per_frame ? "per_frame" : "averaged") << "\n";
    std::cout << "rows:         " << N << "\n";
    std::cout << "classes:      " << K << "\n\n";

    // ---- Class breakdown (count files, not frames, even in per_frame mode) ----
    {
        std::cout << "discovered classes:\n";
        for (int id = 0; id < K; ++id) {
            std::set<int> seen_files;
            for (std::size_t j = 0; j < N; ++j)
                if (cache.labels[j] == id) seen_files.insert(cache.file_ids[j]);
            std::cout << "  " << std::setw(2) << id << "  "
                      << std::setw(8) << std::left << cache.class_names[id]
                      << std::right << "  " << seen_files.size() << " files\n";
        }
        std::cout << "\n";
    }

    // ---- Train/test split (always group-aware; identical to a regular
    //      column split when each file has exactly one row) ----
    Matrix<float>    Y = one_hot<float>(cache.labels, K);
    auto split = group_train_test_split(cache.X, Y, cache.file_ids, 0.2, /*seed=*/1);

    std::cout << "split: " << split.X_train.cols() << " train, "
              << split.X_test.cols()  << " test "
              << (cache.per_frame ? "frames" : "samples")
              << "  (seed=1, file-aware)\n";

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
    const int         epochs     = 50;

    std::cout << "loss:       cross-entropy\n"
              << "optimiser:  Adam (lr=1e-3)\n"
              << "batch:      " << batch_size << "\n"
              << "epochs:     " << epochs << "\n\n";

    // ---- Training loop ----
    const std::string acc_label = cache.per_frame ? "test acc (per file)" : "test acc";
    std::cout << "epoch  train loss  train acc   test loss   " << acc_label << "   time\n";
    std::cout << "-----  ----------  ---------   ---------   "
              << std::string(acc_label.size(), '-') << "   -----\n";

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
        auto [test_loss, test_acc] = evaluate(net, ce, X_te, split.Y_test,
                                               split.file_ids_test,
                                               cache.per_frame, batch_size);

        double secs = std::chrono::duration<double>(clock::now() - t_epoch).count();

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(5)  << epoch        << "  "
                  << std::setw(10) << train_loss   << "  "
                  << std::setw(9)  << train_acc    << "   "
                  << std::setw(9)  << test_loss    << "   "
                  << std::setw(static_cast<int>(acc_label.size())) << test_acc << "   "
                  << std::setprecision(1) << std::setw(4) << secs << "s"
                  << "\n" << std::flush;
    }

    double total_secs = std::chrono::duration<double>(clock::now() - t_start).count();

    // ---- Final report + confusion matrix ----
    net.eval();
    auto [final_loss, final_acc] = evaluate(net, ce, X_te, split.Y_test,
                                             split.file_ids_test,
                                             cache.per_frame, batch_size);

    std::cout << "\ntotal training time: " << std::setprecision(1)
              << total_secs << "s\n"
              << std::setprecision(4)
              << "final test accuracy: " << final_acc << "\n";

    auto conf = build_confusion(net, X_te, split.Y_test, split.file_ids_test,
                                 cache.per_frame, K);
    print_confusion(conf, cache.class_names);

    return 0;
}
