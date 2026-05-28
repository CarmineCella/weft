// test_feature_cache.cpp
//
// Tests for the feature-dataset save/load binary format (v2).
//
#include "FeatureCache.h"
#include "Matrix.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using weft::CachedFeatures;
using weft::Matrix;
using weft::load_features;
using weft::save_features;

static int g_run = 0, g_failed = 0;
static void check(bool cond, const std::string& name) {
    ++g_run;
    std::cout << (cond ? "  [ ok ] " : "  [FAIL] ") << name << '\n';
    if (!cond) ++g_failed;
}
static bool close(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) < eps;
}

int main() {
    std::cout << "weft :: FeatureCache tests\n";

    const std::string path = "/tmp/weft_test.feat";

    // ---- 1. Round trip preserves everything (averaged mode, per_frame=false) ----
    {
        CachedFeatures orig;
        orig.X = Matrix<float>{
            {1.0f,  2.5f,  3.0f, -1.5f, 0.0f},
            {4.0f,  5.0f,  6.0f,  7.0f, 8.0f}
        };
        orig.labels       = {0, 1, 0, 2, 1};
        orig.class_names  = {"alpha", "beta", "gamma"};
        orig.feature_type = "mfcc";
        orig.per_frame    = false;
        orig.file_ids     = {0, 1, 2, 3, 4};   // one file per sample

        save_features(path, orig);
        CachedFeatures got = load_features(path);
        std::remove(path.c_str());

        check(got.X.rows() == 2,                  "averaged: feature_dim");
        check(got.X.cols() == 5,                  "averaged: n_samples");
        check(got.class_names.size() == 3,        "averaged: n_classes");
        check(got.feature_type == "mfcc",         "averaged: feature_type");
        check(got.per_frame == false,             "averaged: per_frame flag");

        bool labels_ok = (got.labels == std::vector<int>{0, 1, 0, 2, 1});
        check(labels_ok,                          "averaged: labels");

        bool ids_ok = (got.file_ids == std::vector<int>{0, 1, 2, 3, 4});
        check(ids_ok,                             "averaged: file_ids");

        bool names_ok = (got.class_names ==
                         std::vector<std::string>{"alpha", "beta", "gamma"});
        check(names_ok,                           "averaged: class_names");

        bool matrix_ok = true;
        for (std::size_t i = 0; i < 2 && matrix_ok; ++i)
            for (std::size_t j = 0; j < 5 && matrix_ok; ++j)
                if (!close(got.X(i, j), orig.X(i, j))) matrix_ok = false;
        check(matrix_ok,                          "averaged: feature matrix");
    }

    // ---- 2. Per-frame round trip (multiple columns share a file_id) ----
    {
        CachedFeatures orig;
        orig.X = Matrix<float>{
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f}
        };  // 2 features x 6 frames
        orig.labels       = {0, 0, 0, 1, 1, 1};
        orig.class_names  = {"alpha", "beta"};
        orig.feature_type = "logmag";
        orig.per_frame    = true;
        orig.file_ids     = {0, 0, 0, 1, 1, 1};   // 3 frames from file 0, 3 from file 1

        save_features(path, orig);
        CachedFeatures got = load_features(path);
        std::remove(path.c_str());

        check(got.per_frame == true,              "per_frame: flag preserved");
        bool ids_ok = (got.file_ids == std::vector<int>{0, 0, 0, 1, 1, 1});
        check(ids_ok,                             "per_frame: file_ids preserved");

        // All frames of file 0 share label, same for file 1
        check(got.labels[0] == 0 && got.labels[1] == 0 && got.labels[2] == 0,
              "per_frame: labels consistent within file");
        check(got.labels[3] == 1 && got.labels[4] == 1 && got.labels[5] == 1,
              "per_frame: labels consistent within file (2)");
    }

    // ---- 3. Larger matrix round trip with extreme values ----
    {
        CachedFeatures orig;
        const std::size_t F = 100, N = 50;
        orig.X = Matrix<float>(F, N);
        for (std::size_t i = 0; i < F; ++i)
            for (std::size_t j = 0; j < N; ++j)
                orig.X(i, j) = std::sin(0.13f * i) * std::cos(0.07f * j) * 1e3f;
        orig.labels.assign(N, 0);
        for (std::size_t j = 0; j < N; ++j) orig.labels[j] = static_cast<int>(j % 5);
        orig.class_names  = {"a", "b", "c", "d", "e"};
        orig.feature_type = "logmag";
        orig.per_frame    = false;
        orig.file_ids.resize(N);
        for (std::size_t j = 0; j < N; ++j) orig.file_ids[j] = static_cast<int>(j);

        save_features(path, orig);
        CachedFeatures got = load_features(path);
        std::remove(path.c_str());

        bool ok = (got.X.rows() == F && got.X.cols() == N);
        for (std::size_t i = 0; i < F && ok; ++i)
            for (std::size_t j = 0; j < N && ok; ++j)
                if (!close(got.X(i, j), orig.X(i, j), 1e-3f)) ok = false;
        check(ok, "larger matrix: 100x50 round trip exact");
    }

    // ---- 4. Empty class names list (degenerate but legal) ----
    {
        CachedFeatures orig;
        orig.X = Matrix<float>(3, 0);
        orig.labels = {};
        orig.class_names = {};
        orig.feature_type = "logmag";
        orig.per_frame = false;
        orig.file_ids = {};

        save_features(path, orig);
        CachedFeatures got = load_features(path);
        std::remove(path.c_str());

        check(got.X.rows() == 3 && got.X.cols() == 0, "empty: zero samples");
        check(got.class_names.empty(),                "empty: no classes");
        check(got.labels.empty(),                     "empty: no labels");
        check(got.file_ids.empty(),                   "empty: no file_ids");
    }

    // ---- 5. Bad magic throws ----
    {
        std::ofstream out(path, std::ios::binary);
        out.write("NOPE", 4);
        for (int i = 0; i < 20; ++i) out.put(0);
        out.close();

        bool threw = false;
        try { load_features(path); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(path.c_str());

        check(threw, "bad magic throws runtime_error");
    }

    // ---- 6. Missing file throws ----
    {
        bool threw = false;
        try { load_features("/tmp/weft_no_such_feat_file.feat"); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "missing file throws runtime_error");
    }

    // ---- 7. Mismatch at save time throws ----
    {
        CachedFeatures bad;
        bad.X = Matrix<float>(2, 5);
        bad.labels = {0, 1, 2};            // 3, not 5
        bad.class_names = {"a", "b", "c"};
        bad.feature_type = "mfcc";
        bad.per_frame = false;
        bad.file_ids = {0, 1, 2, 3, 4};

        bool threw = false;
        try { save_features(path, bad); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(path.c_str());

        check(threw, "save: labels size mismatch throws");
    }

    // ---- 8. file_ids size mismatch at save throws ----
    {
        CachedFeatures bad;
        bad.X = Matrix<float>(2, 5);
        bad.labels = {0, 1, 2, 3, 4};
        bad.class_names = {"a"};
        bad.feature_type = "mfcc";
        bad.per_frame = true;
        bad.file_ids = {0, 0, 1};          // 3, not 5

        bool threw = false;
        try { save_features(path, bad); }
        catch (const std::runtime_error&) { threw = true; }
        std::remove(path.c_str());

        check(threw, "save: file_ids size mismatch throws");
    }

    std::cout << g_run - g_failed << " / " << g_run << " tests passed\n";
    return g_failed == 0 ? 0 : 1;
}
