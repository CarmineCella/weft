// test_data.cpp
//
// Tests for the new data-handling utilities:
//   Matrix::selectColumns, Standardizer, one_hot, train_test_split, accuracy.
//
#include "Matrix.h"
#include "Standardizer.h"
#include "Data.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <set>

using weft::Matrix;
using weft::Standardizer;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}
static bool matClose(const Matrix<float>& A, const Matrix<float>& B,
                     float eps = 1e-4f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            if (!close(A(i, j), B(i, j), eps)) return false;
    return true;
}

int main() {
    std::cout << "weft :: Data tests\n";

    // ---- Matrix::selectColumns ----
    {
        Matrix<float> M{{1, 2, 3, 4}, {5, 6, 7, 8}};
        std::vector<std::size_t> idx = {3, 0, 2};
        Matrix<float> R = M.selectColumns(idx);
        Matrix<float> expected{{4, 1, 3}, {8, 5, 7}};
        check(matClose(R, expected), "selectColumns: picks columns in order");
    }
    {
        Matrix<float> M(2, 2);
        bool threw = false;
        try { M.selectColumns({0, 5}); }
        catch (const std::out_of_range&) { threw = true; }
        check(threw, "selectColumns: out-of-range index throws");
    }

    // ---- Standardizer ----
    {
        Standardizer<float> sc;
        // Feature 0: values 1, 2, 3 -> mean 2, std sqrt(2/3)
        // Feature 1: values 4, 6, 8 -> mean 6, std sqrt(16/3)
        Matrix<float> X{{1, 2, 3}, {4, 6, 8}};
        Matrix<float> Z = sc.fit_transform(X);

        check(close(sc.mean()(0, 0), 2.f) && close(sc.mean()(1, 0), 6.f),
              "Standardizer.fit: per-feature mean");
        check(close(sc.std()(0, 0), std::sqrt(2.f / 3.f)) &&
              close(sc.std()(1, 0), std::sqrt(8.f / 3.f)),
              "Standardizer.fit: per-feature std");

        // After standardising, each row has mean ~0 and std ~1
        for (std::size_t i = 0; i < Z.rows(); ++i) {
            float row_mean = 0, row_var = 0;
            for (std::size_t j = 0; j < Z.cols(); ++j) row_mean += Z(i, j);
            row_mean /= Z.cols();
            for (std::size_t j = 0; j < Z.cols(); ++j)
                row_var += (Z(i, j) - row_mean) * (Z(i, j) - row_mean);
            row_var /= Z.cols();
            check(close(row_mean, 0.f, 1e-5f),
                  "Standardizer.transform: row " + std::to_string(i) + " has mean ~0");
            check(close(row_var, 1.f, 1e-5f),
                  "Standardizer.transform: row " + std::to_string(i) + " has variance ~1");
        }
    }
    {
        // inverse_transform reverses transform
        Standardizer<float> sc;
        Matrix<float> X{{1, 2, 3}, {4, 6, 8}};
        Matrix<float> Z = sc.fit_transform(X);
        Matrix<float> back = sc.inverse_transform(Z);
        check(matClose(back, X, 1e-4f), "Standardizer.inverse_transform: round-trip");
    }
    {
        // Constant feature does not blow up
        Standardizer<float> sc;
        Matrix<float> X{{5, 5, 5}, {1, 2, 3}};
        Matrix<float> Z = sc.fit_transform(X);  // should not produce NaN
        bool finite = true;
        for (std::size_t i = 0; i < Z.rows(); ++i)
            for (std::size_t j = 0; j < Z.cols(); ++j)
                if (!std::isfinite(Z(i, j))) finite = false;
        check(finite, "Standardizer: constant feature does not produce NaN");
    }

    // ---- one_hot ----
    {
        std::vector<int> labels = {0, 2, 1, 2};
        Matrix<float> Y = weft::one_hot<float>(labels, 3);
        Matrix<float> expected{{1, 0, 0, 0},
                               {0, 0, 1, 0},
                               {0, 1, 0, 1}};
        check(matClose(Y, expected), "one_hot: correct one-hot encoding");
    }
    {
        std::vector<int> labels = {0, 5};
        bool threw = false;
        try { weft::one_hot<float>(labels, 3); }
        catch (const std::out_of_range&) { threw = true; }
        check(threw, "one_hot: out-of-range label throws");
    }

    // ---- train_test_split ----
    {
        Matrix<float> X(2, 10);
        for (std::size_t j = 0; j < 10; ++j) {
            X(0, j) = static_cast<float>(j);
            X(1, j) = static_cast<float>(j + 100);
        }
        Matrix<float> Y(1, 10);
        for (std::size_t j = 0; j < 10; ++j) Y(0, j) = static_cast<float>(j);

        auto split = weft::train_test_split(X, Y, 0.3, 42);
        check(split.X_train.cols() == 7 && split.X_test.cols() == 3,
              "train_test_split: 30% test gives 7 train, 3 test");
        check(split.X_train.rows() == 2 && split.X_test.rows() == 2,
              "train_test_split: row counts preserved");

        // No overlap: collected example labels should be {0..9} exactly
        std::set<int> seen;
        for (std::size_t j = 0; j < split.Y_train.cols(); ++j)
            seen.insert(static_cast<int>(split.Y_train(0, j)));
        for (std::size_t j = 0; j < split.Y_test.cols(); ++j)
            seen.insert(static_cast<int>(split.Y_test(0, j)));
        check(seen.size() == 10, "train_test_split: no example appears in both halves");
    }

    // ---- accuracy ----
    {
        //  predictions are softmax probabilities; targets are one-hot.
        //  Argmax of predictions:  0, 1, 1, 2  (using col 0 .. col 3 below)
        //  Argmax of targets:       0, 1, 2, 2
        //  Matches at columns 0, 1, 3  ->  accuracy 3/4.
        Matrix<float> P{{0.7f, 0.1f, 0.2f, 0.1f},
                        {0.2f, 0.8f, 0.5f, 0.3f},
                        {0.1f, 0.1f, 0.3f, 0.6f}};
        Matrix<float> T{{1, 0, 0, 0},
                        {0, 1, 0, 0},
                        {0, 0, 1, 1}};
        float acc = weft::accuracy(P, T);
        check(close(acc, 0.75f), "accuracy: argmax-per-column matching");
    }

    // ---- group_train_test_split ----
    {
        //  6 columns, 3 files (2 frames each).  With test_fraction=0.5
        //  and a fixed seed, expect exactly one entire file (= 2 frames)
        //  in test, the other two files (= 4 frames) in train.
        Matrix<float> X(2, 6);
        Matrix<float> Y(2, 6);
        for (std::size_t j = 0; j < 6; ++j) {
            X(0, j) = static_cast<float>(j);
            X(1, j) = static_cast<float>(j) * 10.0f;
            Y(0, j) = (j < 3) ? 1.0f : 0.0f;
            Y(1, j) = (j < 3) ? 0.0f : 1.0f;
        }
        std::vector<int> file_ids = {0, 0, 1, 1, 2, 2};
        auto g = weft::group_train_test_split(X, Y, file_ids, 0.5, /*seed=*/1);

        check(g.X_train.cols() + g.X_test.cols() == 6,
              "group split: no columns lost");
        check(g.X_test.cols() % 2 == 0,
              "group split: test contains whole files only (2 frames each)");
        check(g.file_ids_test.size() == g.X_test.cols(),
              "group split: file_ids_test length matches X_test cols");

        // No file_id should appear in BOTH train and test.
        std::vector<int> file_ids_train;
        // Recover train file_ids by inverse logic: any file_id not in test.
        bool no_leak = true;
        for (std::size_t j = 0; j < g.X_train.cols(); ++j) {
            // X_train(0, j) is the original column index 0..5; recover file_id
            int orig = static_cast<int>(g.X_train(0, j));
            int fid  = file_ids[orig];
            for (int test_fid : g.file_ids_test)
                if (fid == test_fid) no_leak = false;
        }
        check(no_leak, "group split: no file_id appears in both train and test");
    }

    // ---- group_train_test_split: empty test fraction ----
    {
        Matrix<float> X(2, 4);
        Matrix<float> Y(2, 4);
        std::vector<int> file_ids = {0, 0, 1, 1};
        auto g = weft::group_train_test_split(X, Y, file_ids, 0.0, /*seed=*/1);
        check(g.X_test.cols() == 0 && g.X_train.cols() == 4,
              "group split: test_fraction=0 puts all in train");
    }

    std::cout << "\n" << (g_run - g_failed) << " / " << g_run
              << " checks passed.\n";
    return g_failed == 0 ? 0 : 1;
}
