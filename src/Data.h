#pragma once
//
// Data.h  --  free-function utilities for data handling.  Reusable across
//             every example (IRIS, MNIST, audio, autoencoder, ...).
//
//   shuffled_indices(n, seed)     : return [0..n) randomly permuted.
//   one_hot<T>(labels, K)         : (K x N) one-hot matrix from class labels.
//   train_test_split<T>(X, Y, ...): random split into train/test.
//   group_train_test_split<T>(...): split by file_id, no leakage across groups.
//   accuracy<T>(predictions, targets)
//                                 : argmax-per-column match rate.
//
#include "Matrix.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace weft {

// ---- index shuffling ------------------------------------------------------
inline std::vector<std::size_t> shuffled_indices(std::size_t n, unsigned seed) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937(seed));
    return idx;
}

// ---- one-hot encoding ----------------------------------------------------
//   labels[j] in [0, num_classes).  Result Y is (num_classes x N) with
//   Y[labels[j], j] = 1 and all other entries 0.
template <typename T = float>
Matrix<T> one_hot(const std::vector<int>& labels, std::size_t num_classes) {
    Matrix<T> Y(num_classes, labels.size(), T(0));
    for (std::size_t j = 0; j < labels.size(); ++j) {
        const int c = labels[j];
        if (c < 0 || static_cast<std::size_t>(c) >= num_classes)
            throw std::out_of_range("one_hot: class label out of range");
        Y(c, j) = T(1);
    }
    return Y;
}

// ---- train/test split ----------------------------------------------------
template <typename T>
struct Split {
    Matrix<T> X_train, Y_train;
    Matrix<T> X_test,  Y_test;
};

template <typename T = float>
Split<T> train_test_split(const Matrix<T>& X, const Matrix<T>& Y,
                          double test_fraction, unsigned seed) {
    if (X.cols() != Y.cols())
        throw std::invalid_argument(
            "train_test_split: X and Y must have the same number of columns");
    if (test_fraction < 0.0 || test_fraction > 1.0)
        throw std::invalid_argument("train_test_split: test_fraction not in [0, 1]");

    const std::size_t N      = X.cols();
    const std::size_t n_test = static_cast<std::size_t>(test_fraction * N);
    const std::size_t n_tr   = N - n_test;

    auto idx = shuffled_indices(N, seed);
    std::vector<std::size_t> tr(idx.begin(), idx.begin() + n_tr);
    std::vector<std::size_t> te(idx.begin() + n_tr, idx.end());

    Split<T> s;
    s.X_train = X.selectColumns(tr);
    s.Y_train = Y.selectColumns(tr);
    s.X_test  = X.selectColumns(te);
    s.Y_test  = Y.selectColumns(te);
    return s;
}

// ---- group-aware train/test split ---------------------------------------
//   Splits columns by GROUP (e.g. file_id) rather than by individual column.
//   When a dataset has multiple columns per source file (e.g. per-frame audio
//   features), splitting columns at random would leak frames from the same
//   file into both train and test, inflating test accuracy.  This variant
//   shuffles the unique group ids and assigns whole groups to train or test.
//
//   Returns the same X_train/Y_train/X_test/Y_test as Split, plus the
//   file_ids that remained in the test set (so the classifier can aggregate
//   predictions across all frames of the same file at eval time).
template <typename T>
struct GroupedSplit {
    Matrix<T>        X_train, Y_train;
    Matrix<T>        X_test,  Y_test;
    std::vector<int> file_ids_test;     // file id for each column of X_test
};

template <typename T = float>
GroupedSplit<T> group_train_test_split(const Matrix<T>& X, const Matrix<T>& Y,
                                        const std::vector<int>& file_ids,
                                        double test_fraction, unsigned seed)
{
    if (X.cols() != Y.cols())
        throw std::invalid_argument(
            "group_train_test_split: X and Y must have the same number of columns");
    if (X.cols() != file_ids.size())
        throw std::invalid_argument(
            "group_train_test_split: file_ids length must match X.cols()");
    if (test_fraction < 0.0 || test_fraction > 1.0)
        throw std::invalid_argument(
            "group_train_test_split: test_fraction not in [0, 1]");

    // Unique file ids, in order of first appearance.
    std::vector<int> unique_ids;
    {
        std::vector<bool> seen;
        for (int id : file_ids) {
            if (id < 0)
                throw std::invalid_argument(
                    "group_train_test_split: negative file_id");
            if (static_cast<std::size_t>(id) >= seen.size())
                seen.resize(id + 1, false);
            if (!seen[id]) {
                seen[id] = true;
                unique_ids.push_back(id);
            }
        }
    }

    // Shuffle and split the unique ids.
    std::shuffle(unique_ids.begin(), unique_ids.end(), std::mt19937(seed));
    const std::size_t n_test_files = static_cast<std::size_t>(
        test_fraction * unique_ids.size());

    // Mark each id as test or train using a flat lookup table for speed.
    std::vector<bool> is_test(
        *std::max_element(file_ids.begin(), file_ids.end()) + 1, false);
    for (std::size_t i = 0; i < n_test_files; ++i)
        is_test[unique_ids[i]] = true;

    // Bucket the column indices accordingly.
    std::vector<std::size_t> tr_cols, te_cols;
    tr_cols.reserve(file_ids.size());
    te_cols.reserve(file_ids.size());
    for (std::size_t j = 0; j < file_ids.size(); ++j) {
        if (is_test[file_ids[j]]) te_cols.push_back(j);
        else                      tr_cols.push_back(j);
    }

    GroupedSplit<T> s;
    s.X_train = X.selectColumns(tr_cols);
    s.Y_train = Y.selectColumns(tr_cols);
    s.X_test  = X.selectColumns(te_cols);
    s.Y_test  = Y.selectColumns(te_cols);
    s.file_ids_test.reserve(te_cols.size());
    for (auto c : te_cols) s.file_ids_test.push_back(file_ids[c]);
    return s;
}

// ---- classification accuracy ---------------------------------------------
//   Compares argmax-per-column of predictions to argmax-per-column of targets.
//   Works whether `targets` is one-hot or any monotone score; whether
//   `predictions` is post-softmax probabilities or raw logits.
template <typename T = float>
T accuracy(const Matrix<T>& predictions, const Matrix<T>& targets) {
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols())
        throw std::invalid_argument("accuracy: shape mismatch");

    const std::size_t R = predictions.rows();
    const std::size_t N = predictions.cols();
    std::size_t correct = 0;
    for (std::size_t j = 0; j < N; ++j) {
        std::size_t pred_class = 0, true_class = 0;
        for (std::size_t i = 1; i < R; ++i) {
            if (predictions(i, j) > predictions(pred_class, j)) pred_class = i;
            if (targets    (i, j) > targets    (true_class, j)) true_class = i;
        }
        if (pred_class == true_class) ++correct;
    }
    return static_cast<T>(correct) / static_cast<T>(N);
}

} // namespace weft
