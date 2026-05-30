#pragma once
//
// Embedding.h  --  map integer token indices to learnable vectors.
//
// An embedding table is a (d_model x vocab_size) matrix; the embedding
// of token id k is the k-th column.  Forward is a lookup; backward is
// a scatter-add of the upstream gradient into the columns that were
// looked up.
//
// We don't make this a Layer<T>: the Layer interface promises a
// Matrix<T> input, but token ids are integers and casting them through
// a float matrix is awkward (NaN-prone, semantically wrong).  Instead
// Embedding is a small standalone object with its own forward (taking
// std::vector<int>) and backward (taking the upstream gradient
// Matrix<T>).  It still cooperates with Optimizer<T> via the same
// step(parameter, grad) interface every other layer uses.
//
// Parameter init: small-stddev Gaussian (1 / sqrt(d_model)).  The
// transformer's residual stream prefers small initial values; He
// initialisation would be too aggressive for a lookup table that has
// no fan_in-style structure.
//
#include "Matrix.h"
#include "Optimizer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

template <typename T = float>
class Embedding {
public:
    Embedding(std::size_t vocab_size, std::size_t d_model, unsigned seed = 1)
        : vocab_size_(vocab_size), d_model_(d_model),
          W_(d_model, vocab_size), dW_(d_model, vocab_size, T(0))
    {
        // Small Gaussian; the residual stream's initial activations are
        // additive, so larger values would blow up early gradient
        // magnitudes.  1/sqrt(d_model) is a standard transformer choice.
        const T stddev = T(1) / std::sqrt(static_cast<T>(d_model));
        W_.randomizeNormal(T(0), stddev, seed);
    }

    // tokens: integer ids in [0, vocab_size).  Returns the embedded
    // sequence as (d_model x seq_len) -- one column per token, ready to
    // feed the rest of the transformer.
    Matrix<T> forward(const std::vector<int>& tokens) {
        last_tokens_ = tokens;
        const std::size_t L = tokens.size();
        Matrix<T> Y(d_model_, L);
        for (std::size_t t = 0; t < L; ++t) {
            const int id = tokens[t];
            if (id < 0 || static_cast<std::size_t>(id) >= vocab_size_)
                throw std::out_of_range(
                    "Embedding::forward: token id " + std::to_string(id)
                    + " out of vocab [0, " + std::to_string(vocab_size_) + ")");
            for (std::size_t d = 0; d < d_model_; ++d)
                Y(d, t) = W_(d, static_cast<std::size_t>(id));
        }
        return Y;
    }

    // dY: (d_model x seq_len) upstream gradient.  Scatter into dW_ at
    // the columns we looked up.  No "input gradient" to return: the
    // input was a vector of integer indices, not differentiable.
    //
    // Accumulates if the same token appears multiple times in the
    // sequence (correct behaviour: the table column contributed once
    // per occurrence, so the gradient sums).
    void backward(const Matrix<T>& dY) {
        if (dY.cols() != last_tokens_.size() || dY.rows() != d_model_)
            throw std::invalid_argument(
                "Embedding::backward: gradient shape doesn't match the last forward");
        // Reset dW_ before accumulating this step's contributions.
        for (std::size_t k = 0; k < vocab_size_; ++k)
            for (std::size_t d = 0; d < d_model_; ++d)
                dW_(d, k) = T(0);
        for (std::size_t t = 0; t < last_tokens_.size(); ++t) {
            const std::size_t id = static_cast<std::size_t>(last_tokens_[t]);
            for (std::size_t d = 0; d < d_model_; ++d)
                dW_(d, id) += dY(d, t);
        }
    }

    void update(Optimizer<T>& opt) { opt.step(W_, dW_); }

    void save_params(std::ostream& out) const {
        const std::uint32_t r = static_cast<std::uint32_t>(W_.rows());
        const std::uint32_t c = static_cast<std::uint32_t>(W_.cols());
        out.write(reinterpret_cast<const char*>(&r), sizeof(r));
        out.write(reinterpret_cast<const char*>(&c), sizeof(c));
        for (std::size_t i = 0; i < W_.rows(); ++i)
            for (std::size_t j = 0; j < W_.cols(); ++j) {
                const T v = W_(i, j);
                out.write(reinterpret_cast<const char*>(&v), sizeof(T));
            }
    }
    void load_params(std::istream& in) {
        std::uint32_t r = 0, c = 0;
        in.read(reinterpret_cast<char*>(&r), sizeof(r));
        in.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (r != W_.rows() || c != W_.cols())
            throw std::runtime_error(
                "Embedding::load_params: shape mismatch (build the same "
                "architecture before loading)");
        for (std::size_t i = 0; i < W_.rows(); ++i)
            for (std::size_t j = 0; j < W_.cols(); ++j) {
                T v = T(0);
                in.read(reinterpret_cast<char*>(&v), sizeof(T));
                W_(i, j) = v;
            }
    }

    std::size_t vocab_size() const { return vocab_size_; }
    std::size_t d_model()    const { return d_model_;    }
    const Matrix<T>& W()     const { return W_; }
    const Matrix<T>& dW()    const { return dW_; }

private:
    std::size_t      vocab_size_, d_model_;
    Matrix<T>        W_, dW_;
    std::vector<int> last_tokens_;
};

} // namespace weft
