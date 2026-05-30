#pragma once
//
// TransformerEncoderBlock.h  --  one full pre-LN encoder block:
//                                attention + FFN + residual + LayerNorm.
//
// Composition (pre-LN variant, standard for modern transformers
// because it's more stable to train):
//
//   X --+--> LayerNorm --> MultiHeadAttention --+--> Y
//       |                                       |
//       +-----------------------(residual)------+
//
//   Y --+--> LayerNorm --> Dense -> ReLU -> Dense --+--> Z
//       |                                            |
//       +-----------------------(residual)-----------+
//
// The feed-forward sub-layer is the standard two-Dense bottleneck-
// outside design: d_model -> d_ff -> d_model, with d_ff usually 4 *
// d_model in real transformers (we default to that here but expose it
// as a constructor argument so a small toy block can use a smaller
// expansion).
//
// All sub-layers manage their own forward caches and backward state;
// the only thing this block has to do beyond chaining them is add the
// residual at the right point and route the gradient back through it.
// In particular, the residual means each input is reached by *two*
// paths: directly (identity) and through the sub-stack.  Their
// gradients sum.
//
// One TransformerEncoderBlock has all four projections of one MHA plus
// gamma/beta of two LayerNorms plus two Dense weights and biases.  All
// learnable; update() walks them in fixed order so save/load is
// trivially correct.
//
#include "Dense.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Matrix.h"
#include "MultiHeadAttention.h"
#include "ReLU.h"

#include <cstddef>
#include <string>

namespace weft {

template <typename T = float>
class TransformerEncoderBlock : public Layer<T> {
public:
    // d_model    feature dimension (same on input and output)
    // n_heads    attention heads; d_model must be divisible by it
    // d_ff       inner dim of the feed-forward sublayer (default 4*d_model)
    TransformerEncoderBlock(std::size_t d_model,
                            std::size_t n_heads,
                            std::size_t d_ff = 0)
        : d_model_(d_model), n_heads_(n_heads),
          d_ff_(d_ff > 0 ? d_ff : 4 * d_model),
          ln1_(d_model), mha_(d_model, n_heads),
          ln2_(d_model),
          ff1_(d_model, d_ff_ > 0 ? d_ff_ : 4 * d_model),
          ff2_(d_ff_ > 0 ? d_ff_ : 4 * d_model, d_model)
    {}

    Matrix<T> forward(const Matrix<T>& X) override {
        X_cache_ = X;                                  // for residual backward

        // ---- attention sublayer ----
        Matrix<T> a = ln1_.forward(X);                 // LN(X)
        Matrix<T> b = mha_.forward(a);                 // Attn(LN(X))
        Matrix<T> Y = X + b;                           // residual

        Y_cache_ = Y;                                  // for second residual backward

        // ---- feed-forward sublayer ----
        Matrix<T> c = ln2_.forward(Y);                 // LN(Y)
        Matrix<T> d = ff1_.forward(c);                 // Dense
        Matrix<T> e = relu_.forward(d);                // ReLU
        Matrix<T> f = ff2_.forward(e);                 // Dense
        return Y + f;                                  // residual
    }

    Matrix<T> backward(const Matrix<T>& dZ) override {
        // ---- feed-forward backward ----
        // Z = Y + f, so f gets dZ and Y picks up dZ from the residual
        // plus whatever comes through the sub-stack.
        Matrix<T> df  = dZ;
        Matrix<T> de  = ff2_.backward(df);
        Matrix<T> dd  = relu_.backward(de);
        Matrix<T> dc  = ff1_.backward(dd);
        Matrix<T> dY_from_ffn = ln2_.backward(dc);
        Matrix<T> dY  = dZ + dY_from_ffn;              // residual sum

        // ---- attention backward ----
        // Y = X + b, so b gets dY and X picks up dY from the residual
        // plus whatever comes through the attention sub-stack.
        Matrix<T> db  = dY;
        Matrix<T> da  = mha_.backward(db);
        Matrix<T> dX_from_attn = ln1_.backward(da);
        return dY + dX_from_attn;                      // residual sum
    }

    void update(Optimizer<T>& opt) override {
        ln1_.update(opt);
        mha_.update(opt);
        ln2_.update(opt);
        ff1_.update(opt);
        ff2_.update(opt);
    }

    std::string describe() const override {
        return "TransformerEncoderBlock(d_model="
             + std::to_string(d_model_)
             + ", n_heads=" + std::to_string(n_heads_)
             + ", d_ff=" + std::to_string(d_ff_) + ")";
    }

    // save/load in a fixed order so the architecture rebuild can
    // reproduce the state exactly.  Each sub-layer handles its own
    // bytes.
    void save_params(std::ostream& out) const override {
        ln1_.save_params(out);
        mha_.save_params(out);
        ln2_.save_params(out);
        ff1_.save_params(out);
        ff2_.save_params(out);
    }
    void load_params(std::istream& in) override {
        ln1_.load_params(in);
        mha_.load_params(in);
        ln2_.load_params(in);
        ff1_.load_params(in);
        ff2_.load_params(in);
    }

    std::size_t d_model() const { return d_model_; }
    std::size_t n_heads() const { return n_heads_; }
    std::size_t d_ff()    const { return d_ff_;    }

private:
    std::size_t d_model_, n_heads_, d_ff_;

    LayerNorm<T>            ln1_;
    MultiHeadAttention<T>   mha_;
    LayerNorm<T>            ln2_;
    Dense<T>                ff1_;
    ReLU<T>                 relu_;
    Dense<T>                ff2_;

    Matrix<T> X_cache_, Y_cache_;    // unused after construction, but kept
                                     // alive in case future variants need them
};

} // namespace weft
