#pragma once
//
// Matrix.h  --  foundation of "weft" (Weights Estimated From Training).
//
// A small, dependency-free (standard library only), header-only,
// templated dense matrix class.
//
// CONVENTIONS USED THROUGHOUT THE LIBRARY
// ---------------------------------------
//   * Math convention: a batch of examples is stored with one example
//     PER COLUMN.  So an input batch has shape (features x batch_size),
//     a weight matrix has shape (out_features x in_features), and a
//     layer's forward pass is:   Z = W * X + b
//   * Storage convention: numbers are kept in a single flat std::vector
//     in ROW-MAJOR order, i.e. element (i, j) lives at data_[i*cols_ + j].
//     (Storage layout is independent of the math convention above.)
//
// The scalar type T is a template parameter (use float for speed/memory,
// double when you want to numerically check gradients later).
//

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <iomanip>

namespace weft {

template <typename T = float>
class Matrix {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------
    Matrix() : rows_(0), cols_(0) {}

    // rows x cols, every element initialised to `value` (default 0).
    Matrix(std::size_t rows, std::size_t cols, T value = T(0))
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    // Build from a nested initializer list, e.g.  Matrix<float> A{{1,2},{3,4}};
    // Handy for writing small tests by hand.
    Matrix(std::initializer_list<std::initializer_list<T>> init) {
        rows_ = init.size();
        cols_ = rows_ ? init.begin()->size() : 0;
        data_.reserve(rows_ * cols_);
        for (const auto& row : init) {
            if (row.size() != cols_)
                throw std::invalid_argument("Matrix: ragged initializer list");
            for (const T& v : row) data_.push_back(v);
        }
    }

    // ------------------------------------------------------------------
    // Shape
    // ------------------------------------------------------------------
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    // ------------------------------------------------------------------
    // Element access (row-major).  Two versions so it works on both
    // mutable and const matrices:   A(i,j) = 3;   x = A(i,j);
    // ------------------------------------------------------------------
    T&       operator()(std::size_t i, std::size_t j)       { return data_[i * cols_ + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    // ------------------------------------------------------------------
    // Fill / random initialisation
    // ------------------------------------------------------------------
    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

    // Gaussian init -- the usual starting point for weights.
    void randomizeNormal(T mean = T(0), T stddev = T(1),
                         unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        std::normal_distribution<T> dist(mean, stddev);
        for (T& x : data_) x = dist(gen);
    }

    void randomizeUniform(T lo = T(-1), T hi = T(1),
                          unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<T> dist(lo, hi);
        for (T& x : data_) x = dist(gen);
    }

    // ------------------------------------------------------------------
    // Transpose.  Needed constantly in backprop.
    // ------------------------------------------------------------------
    Matrix transpose() const {
        Matrix R(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                R(j, i) = (*this)(i, j);
        return R;
    }

    // ------------------------------------------------------------------
    // Apply an element-wise function. This is how activation functions
    // (and their derivatives) will be implemented.  Templated on the
    // callable so we pay no std::function overhead.
    // ------------------------------------------------------------------
    template <typename Func>
    Matrix apply(Func f) const {
        Matrix R(rows_, cols_);
        for (std::size_t k = 0; k < data_.size(); ++k) R.data_[k] = f(data_[k]);
        return R;
    }

    template <typename Func>
    void applyInPlace(Func f) {
        for (T& x : data_) x = f(x);
    }

    // ------------------------------------------------------------------
    // Reduction: sum across columns -> a (rows_ x 1) column vector.
    // Because each column is one example, this sums OVER THE BATCH.
    // (This is exactly how we will accumulate the bias gradient.)
    // ------------------------------------------------------------------
    Matrix sumColumns() const {
        Matrix R(rows_, 1, T(0));
        for (std::size_t i = 0; i < rows_; ++i) {
            T s = T(0);
            for (std::size_t j = 0; j < cols_; ++j) s += (*this)(i, j);
            R(i, 0) = s;
        }
        return R;
    }

    // ------------------------------------------------------------------
    // Matrix multiplication:  (m x k) * (k x n) -> (m x n)
    // The i-k-j loop ordering keeps the innermost loop walking
    // contiguously through memory, which is much friendlier to the CPU
    // cache than the textbook i-j-k ordering.
    // ------------------------------------------------------------------
    Matrix operator*(const Matrix& B) const {
        if (cols_ != B.rows_)
            throw std::invalid_argument("Matrix*: inner dimensions do not match");
        Matrix C(rows_, B.cols_, T(0));
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t k = 0; k < cols_; ++k) {
                const T a = (*this)(i, k);
                for (std::size_t j = 0; j < B.cols_; ++j)
                    C(i, j) += a * B(k, j);
            }
        }
        return C;
    }

    // ------------------------------------------------------------------
    // Scalar operations
    // ------------------------------------------------------------------
    Matrix operator*(T s) const { return apply([s](T x){ return x * s; }); }
    Matrix operator/(T s) const { return apply([s](T x){ return x / s; }); }

    // ------------------------------------------------------------------
    // Element-wise + and -, WITH column broadcasting.
    //   * if both shapes match  -> plain element-wise op
    //   * if the right operand is a column vector (cols == 1) whose rows
    //     match -> it is broadcast across every column.
    // The second case is what lets us write  W*X + b  where b is (out x 1).
    // ------------------------------------------------------------------
    Matrix operator+(const Matrix& B) const { return elementwise(B, [](T a, T b){ return a + b; }); }
    Matrix operator-(const Matrix& B) const { return elementwise(B, [](T a, T b){ return a - b; }); }

    Matrix& operator+=(const Matrix& B) { *this = *this + B; return *this; }
    Matrix& operator-=(const Matrix& B) { *this = *this - B; return *this; }

    // ------------------------------------------------------------------
    // Debug printing:   std::cout << M;
    // ------------------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const Matrix& M) {
        os << "Matrix(" << M.rows_ << "x" << M.cols_ << ")\n";
        for (std::size_t i = 0; i < M.rows_; ++i) {
            for (std::size_t j = 0; j < M.cols_; ++j)
                os << std::setw(10) << std::setprecision(4) << M(i, j) << ' ';
            os << '\n';
        }
        return os;
    }

private:
    std::size_t rows_, cols_;
    std::vector<T> data_;

    // Shared implementation for + and - including the broadcast rule.
    template <typename Op>
    Matrix elementwise(const Matrix& B, Op op) const {
        if (B.rows_ == rows_ && B.cols_ == cols_) {            // same shape
            Matrix R(rows_, cols_);
            for (std::size_t k = 0; k < data_.size(); ++k)
                R.data_[k] = op(data_[k], B.data_[k]);
            return R;
        }
        if (B.cols_ == 1 && B.rows_ == rows_) {                // broadcast column
            Matrix R(rows_, cols_);
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < cols_; ++j)
                    R(i, j) = op((*this)(i, j), B(i, 0));
            return R;
        }
        throw std::invalid_argument("elementwise: shape mismatch (no valid broadcast)");
    }
};

// ----------------------------------------------------------------------
// Free functions
// ----------------------------------------------------------------------

// scalar * Matrix  (so both  2.0f * M  and  M * 2.0f  work)
template <typename T>
Matrix<T> operator*(T s, const Matrix<T>& M) { return M * s; }

// Hadamard (element-wise) product.  operator* is taken by matmul, so the
// element-wise product gets its own name.  Used heavily in backprop.
template <typename T>
Matrix<T> hadamard(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("hadamard: shape mismatch");
    Matrix<T> R(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            R(i, j) = A(i, j) * B(i, j);
    return R;
}

} // namespace weft
