#include "Data.h"

#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

static int failed = 0;
static void check(bool ok, const std::string& name) {
    std::cout << (ok ? "[ ok ] " : "[FAIL] ") << name << '\n';
    if (!ok) ++failed;
}
template <typename F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
int main() {
    using namespace weft;
    Matrix<float> x(2, 20), y(1, 20);
    for (std::size_t i = 0; i < 20; ++i) { x(0, i) = float(i); y(0, i) = float(i); }
    const auto split = train_validation_test_split(x, y, 0.2, 0.2, 4);
    check(split.X_train.cols() == 12 && split.X_validation.cols() == 4 && split.X_test.cols() == 4,
          "three-way split reserves separate validation and test data");
    std::set<int> seen;
    bool paired = true;
    for (const auto* part : {&split.X_train, &split.X_validation, &split.X_test})
        for (std::size_t j = 0; j < part->cols(); ++j)
            paired &= seen.insert(int((*part)(0, j))).second;
    for (std::size_t j = 0; j < split.X_validation.cols(); ++j)
        paired &= split.X_validation(0, j) == split.Y_validation(0, j);
    check(paired && seen.size() == 20, "no duplicates, omissions, or detached validation targets");
    const auto repeated = train_validation_test_split(x, y, 0.2, 0.2, 4);
    bool reproducible = true;
    for (std::size_t j = 0; j < split.X_train.cols(); ++j)
        reproducible &= split.X_train(0, j) == repeated.X_train(0, j);
    check(reproducible, "split membership is reproducible with a fixed seed");
    Matrix<float> grouped_x(1, 6), grouped_y(1, 6);
    const std::vector<int> ids{1000000000, 1000000000, 7, 7, 9, 9};
    const auto grouped = group_train_test_split(grouped_x, grouped_y, ids, 0.34, 3);
    std::set<int> train_ids(grouped.file_ids_train.begin(), grouped.file_ids_train.end());
    bool disjoint = true;
    for (int id : grouped.file_ids_test) disjoint &= !train_ids.count(id);
    check(disjoint && grouped.file_ids_train.size() == 4 && grouped.file_ids_test.size() == 2,
          "sparse group IDs stay intact without huge ID-indexed allocations");
    const auto empty = group_train_test_split(Matrix<float>(2, 0), Matrix<float>(1, 0), {}, 0.2, 1);
    check(empty.X_train.cols() == 0 && empty.X_test.cols() == 0, "empty grouped input is defined");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(throws([&] { train_test_split(x, y, nan, 1); }) &&
          throws([&] { group_train_test_split(grouped_x, grouped_y, ids, nan, 1); }) &&
          throws([&] { train_validation_test_split(x, y, 0.8, 0.4, 1); }),
          "invalid split fractions are rejected");
    return failed ? 1 : 0;
}
