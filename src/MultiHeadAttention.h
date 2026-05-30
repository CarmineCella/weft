#pragma once
//
// MultiHeadAttention.h  --  multi-head self-attention as a Layer<T>.
//
// Architecturally:
//
//   X   (d_model x seq_len)               <-- one sequence, one token per column
//    +---> Wq -> Q -+
//    +---> Wk -> K -+----> split into n_heads --> per-head SDPA --> concat --> Wo --> Y
//    +---> Wv -> V -+
//                                                 (each head sees d_head = d_model / n_heads
//                                                  features along the row axis)
//
// The projections Wq, Wk, Wv, Wo are full Dense layers (with biases),
// because the "examples are columns" convention treats each token in the
// sequence as one example, exactly what Dense expects.  So we get He
// initialisation, save/load and the optimizer hook for free; the only
// new code in this file is the per-head split/join and the routing of
// the three-input/three-output gradients through the four Dense layers.
//
// Backward routing.  Y = Wo @ concat(head_h(Qh, Kh, Vh) for h).  Given
// dY, we run it through Wo.backward to get d(concat), split that by
// head to get dYh per head, ask each head's SDPA for (dQh, dKh, dVh),
// rejoin to (dQ, dK, dV), and then pass each through its Dense.backward
// to get the three contributions to dX.  These three are SUMMED because
// in self-attention X is the *common* input to all three projections,
// so its gradient is the sum of how it influenced Q, K and V.
//
// One head per SDPA instance (each owns its own Softmax cache); this
// makes the per-head backward independent and correct.
//
#include "Dense.h"
#include "Layer.h"
#include "Matrix.h"
#include "ScaledDotProductAttention.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

template <typename T = float>
class MultiHeadAttention : public Layer<T> {
public:
    // d_model    full feature dimension (must equal across Q, K, V, output)
    // n_heads    number of attention heads (d_model must be divisible by n_heads)
    MultiHeadAttention(std::size_t d_model, std::size_t n_heads)
        : d_model_(d_model),
          n_heads_(n_heads),
          d_head_(validate(d_model, n_heads)),     // throws BEFORE the divide-by-zero
          Wq_(d_model, d_model),
          Wk_(d_model, d_model),
          Wv_(d_model, d_model),
          Wo_(d_model, d_model),
          heads_(n_heads)
    {}

    // X (d_model x seq_len)  ->  Y (d_model x seq_len)
    Matrix<T> forward(const Matrix<T>& X) override {
        if (X.rows() != d_model_)
            throw std::invalid_argument(
                "MultiHeadAttention::forward: input rows must equal d_model");

        // Step 1: linear projections of X -- one column per token.
        Matrix<T> Q = Wq_.forward(X);
        Matrix<T> K = Wk_.forward(X);
        Matrix<T> V = Wv_.forward(X);

        // Step 2: split feature axis into heads, run attention per head,
        // accumulate outputs into a single concatenated buffer (which
        // becomes the input to the output projection Wo).
        const std::size_t seq_len = X.cols();
        Matrix<T> concat(d_model_, seq_len);
        for (std::size_t h = 0; h < n_heads_; ++h) {
            Matrix<T> Qh = head_slice(Q, h);
            Matrix<T> Kh = head_slice(K, h);
            Matrix<T> Vh = head_slice(V, h);
            Matrix<T> Yh = heads_[h].forward(Qh, Kh, Vh);   // (d_head x seq_len)
            head_join(concat, h, Yh);
        }

        // Step 3: output projection back to d_model.
        return Wo_.forward(concat);
    }

    // dY (d_model x seq_len)  ->  dX (d_model x seq_len)
    Matrix<T> backward(const Matrix<T>& dY) override {
        // Reverse the output projection: gradient w.r.t. the concatenated
        // per-head outputs.
        Matrix<T> dConcat = Wo_.backward(dY);                 // (d_model x seq_len)

        // Per-head backward via each SDPA instance, then re-pack the
        // resulting dQ/dK/dV slices into full d_model-tall matrices for
        // the projection backwards.
        const std::size_t seq_len = dConcat.cols();
        Matrix<T> dQ(d_model_, seq_len);
        Matrix<T> dK(d_model_, seq_len);
        Matrix<T> dV(d_model_, seq_len);
        for (std::size_t h = 0; h < n_heads_; ++h) {
            Matrix<T> dYh = head_slice(dConcat, h);
            auto [dQh, dKh, dVh] = heads_[h].backward(dYh);
            head_join(dQ, h, dQh);
            head_join(dK, h, dKh);
            head_join(dV, h, dVh);
        }

        // Three independent paths back through the projections; X was the
        // common input to all three, so dX is the sum of the contributions.
        Matrix<T> dXq = Wq_.backward(dQ);
        Matrix<T> dXk = Wk_.backward(dK);
        Matrix<T> dXv = Wv_.backward(dV);
        return dXq + dXk + dXv;
    }

    void update(Optimizer<T>& opt) override {
        Wq_.update(opt);
        Wk_.update(opt);
        Wv_.update(opt);
        Wo_.update(opt);
    }

    std::string describe() const override {
        return "MultiHeadAttention(d_model=" + std::to_string(d_model_)
             + ", n_heads=" + std::to_string(n_heads_) + ")";
    }

    void save_params(std::ostream& out) const override {
        Wq_.save_params(out);
        Wk_.save_params(out);
        Wv_.save_params(out);
        Wo_.save_params(out);
    }
    void load_params(std::istream& in) override {
        Wq_.load_params(in);
        Wk_.load_params(in);
        Wv_.load_params(in);
        Wo_.load_params(in);
    }

    std::size_t d_model() const { return d_model_; }
    std::size_t n_heads() const { return n_heads_; }
    std::size_t d_head()  const { return d_head_;  }

private:
    // Rows [h*d_head, (h+1)*d_head) of M, all columns.
    Matrix<T> head_slice(const Matrix<T>& M, std::size_t h) const {
        Matrix<T> out(d_head_, M.cols());
        for (std::size_t i = 0; i < d_head_; ++i)
            for (std::size_t j = 0; j < M.cols(); ++j)
                out(i, j) = M(h * d_head_ + i, j);
        return out;
    }
    // Write `sub` (d_head x seq_len) into rows [h*d_head, (h+1)*d_head) of M.
    void head_join(Matrix<T>& M, std::size_t h, const Matrix<T>& sub) const {
        for (std::size_t i = 0; i < d_head_; ++i)
            for (std::size_t j = 0; j < sub.cols(); ++j)
                M(h * d_head_ + i, j) = sub(i, j);
    }

    std::size_t d_model_, n_heads_, d_head_;
    Dense<T>    Wq_, Wk_, Wv_, Wo_;
    std::vector<ScaledDotProductAttention<T>> heads_;

    // Member-initializer guard: runs before d_head_ is computed so the
    // d_model / n_heads divide can't trap on n_heads == 0.
    static std::size_t validate(std::size_t d_model, std::size_t n_heads) {
        if (n_heads == 0)
            throw std::invalid_argument(
                "MultiHeadAttention: n_heads must be > 0");
        if (d_model % n_heads != 0)
            throw std::invalid_argument(
                "MultiHeadAttention: d_model must be divisible by n_heads");
        return d_model / n_heads;
    }
};

} // namespace weft
