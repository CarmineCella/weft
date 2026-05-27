#pragma once
//
// Data.h  --  free-function utilities for data handling.  Reusable across
//             every example (IRIS, MNIST, audio, autoencoder, ...).
//
//   shuffled_indices(n, seed)     : return [0..n) randomly permuted.
//   one_hot<T>(labels, K)         : (K x N) one-hot matrix from class labels.
//   train_test_split<T>(X, Y, ...): random split into train/test.
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
