#pragma once
//
// SinusoidalPositionalEncoding.h  --  the fixed-pattern position
//                                     encoding from "Attention is All
//                                     You Need" (Vaswani et al., 2017).
//
// Attention is permutation-equivariant on the token axis: shuffle the
// columns of the input and the columns of the output shuffle the same
// way, but each *individual* output column is unchanged.  That means
// the bare attention layer cannot distinguish "the cat sat on the mat"
// from "mat the on sat cat the".  We need to give every position a
// unique fingerprint, which is what this layer does.
//
// Formula (per token position pos, per feature dimension i):
//
//   PE(pos, 2i)     =  sin( pos / 10000^(2i / d_model) )
//   PE(pos, 2i+1)   =  cos( pos / 10000^(2i / d_model) )
//
// Two properties make this choice useful:
//
//   1. Every position has a unique encoding (the geometric frequency
//      progression guarantees no two pos values produce the same vector).
//   2. PE(pos + k) is a *linear* function of PE(pos) for any fixed k,
//      so the network can in principle learn to attend by relative
//      offsets just from the encoding.  See the Vaswani paper, §3.5.
//
// We precompute the full (d_model x max_seq_len) PE matrix at construction
// time and just ADD it column-by-column to the input in forward.  Backward
// is trivial because there are no parameters and the gradient of "add a
// constant" w.r.t. the input is the identity.
//
// This is a Layer<T> with no trainable parameters, so update() is a
// no-op (the default).  Like Dropout and ReLU, it slots into a Network
// transparently.
//
#include "Layer.h"
#include "Matrix.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace weft {

template <typename T = float>
class SinusoidalPositionalEncoding : public Layer<T> {
public:
    // d_model       feature dimension of each token (== input rows)
    // max_seq_len   longest sequence we'll ever encode (forward throws
    //               if asked for a longer one)
    SinusoidalPositionalEncoding(std::size_t d_model, std::size_t max_seq_len)
        : d_model_(d_model), max_seq_len_(max_seq_len),
          PE_(d_model, max_seq_len)
    {
        // Fill PE column by column (pos), inside each column dim by dim.
        // The frequency for the (i, i+1) pair is 1 / 10000^(2 floor(i/2) / d_model).
        // We compute the exponent once per pair to avoid pow() in the inner loop.
        for (std::size_t i = 0; i < d_model; i += 2) {
            const T exponent = static_cast<T>(i) / static_cast<T>(d_model);
            const T inv_freq = std::pow(static_cast<T>(10000), -exponent);
            for (std::size_t pos = 0; pos < max_seq_len; ++pos) {
                const T angle = static_cast<T>(pos) * inv_freq;
                PE_(i, pos) = std::sin(angle);
                if (i + 1 < d_model)
                    PE_(i + 1, pos) = std::cos(angle);
            }
        }
    }

    // X  (d_model x seq_len)  ->  X + PE_[:, :seq_len]
    Matrix<T> forward(const Matrix<T>& X) override {
        if (X.rows() != d_model_)
            throw std::invalid_argument(
                "SinusoidalPositionalEncoding::forward: "
                "input rows must equal d_model");
        if (X.cols() > max_seq_len_)
            throw std::invalid_argument(
                "SinusoidalPositionalEncoding::forward: "
                "sequence longer than max_seq_len; rebuild the layer "
                "with a larger max_seq_len");

        Matrix<T> Y(X.rows(), X.cols());
        for (std::size_t j = 0; j < X.cols(); ++j)
            for (std::size_t i = 0; i < d_model_; ++i)
                Y(i, j) = X(i, j) + PE_(i, j);
        return Y;
    }

    // d(X + PE) / dX = identity, so dX = dY.  No parameter gradients.
    Matrix<T> backward(const Matrix<T>& dY) override {
        return dY;
    }

    std::string describe() const override {
        return "SinusoidalPositionalEncoding(d_model="
             + std::to_string(d_model_)
             + ", max_seq_len=" + std::to_string(max_seq_len_) + ")";
    }

    // No save/load: PE is a deterministic function of d_model and
    // max_seq_len, both of which the caller must specify at
    // construction.  Loading wrong sizes would be a silent error;
    // requiring reconstruction is safer.

    // Inspection: the full PE matrix for plotting or sanity checking.
    const Matrix<T>& positional_encoding() const { return PE_; }
    std::size_t d_model()     const { return d_model_; }
    std::size_t max_seq_len() const { return max_seq_len_; }

private:
    std::size_t d_model_, max_seq_len_;
    Matrix<T>   PE_;
};

} // namespace weft
