#pragma once
//
// TokenTransformer.h  --  a decoder-only transformer over integer tokens:
//                         the model that predicts "what comes next".
//
//   tokens (L integers)
//      |  Embedding(vocab, d_model)             (d_model x L)
//      |  SinusoidalPositionalEncoding
//      |  TransformerEncoderBlock x n_layers    causal mask ON
//      |  LayerNorm
//      |  Dense(d_model -> vocab)
//      v  Softmax                               (vocab x L)
//
// Column t of the output is the model's probability distribution for
// token t+1 given tokens 0..t.  Because every block is causal, one forward
// pass over a sequence yields a prediction at every position at once,
// which is what makes training efficient; generation then calls forward
// repeatedly, each time reading only the last column.
//
// Nothing here knows what the tokens mean.  The same class is trained on
// the notes of four-voice chorales and on the atoms of a matching-pursuit
// decomposition; only the vocabulary size and the tokeniser differ.
//
// All the layers it is made of were built and tested in earlier chapters.
// This file only fixes the order in which they are called, forward and
// backward, and the order in which their parameters are saved.
//
#include "Dense.h"
#include "Checkpoint.h"
#include "Embedding.h"
#include "LayerNorm.h"
#include "Matrix.h"
#include "Optimizer.h"
#include "SinusoidalPositionalEncoding.h"
#include "Softmax.h"
#include "TransformerEncoderBlock.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace weft {

template <typename T = float>
class TokenTransformer {
public:
    // vocab      number of distinct tokens
    // d_model    width of the residual stream
    // n_heads    attention heads per block (must divide d_model)
    // n_layers   number of blocks
    // d_ff       inner width of each block's feed-forward part
    // max_seq    longest sequence forward() will accept
    TokenTransformer(std::size_t vocab, std::size_t d_model, std::size_t n_heads,
                     std::size_t n_layers, std::size_t d_ff, std::size_t max_seq)
        : vocab_(vocab), d_model_(d_model), max_seq_(max_seq),
          n_heads_(n_heads), d_ff_(d_ff > 0 ? d_ff : 4 * d_model),
          embed(vocab, d_model),
          pe(d_model, max_seq),
          final_ln(d_model),
          head(d_model, vocab)
    {
        blocks.reserve(n_layers);
        for (std::size_t i = 0; i < n_layers; ++i) {
            blocks.emplace_back(d_model, n_heads, d_ff_);
            blocks.back().set_causal(true);            // autoregressive
        }
    }

    // Probabilities over the vocabulary at every position, (vocab x L).
    Matrix<T> forward(const std::vector<int>& tokens) {
        return softmax.forward(forward_logits(tokens));
    }

    // Training path for SoftmaxCrossEntropy. No probability-space division.
    Matrix<T> forward_logits(const std::vector<int>& tokens) {
        if (tokens.empty())
            throw std::invalid_argument("TokenTransformer: empty token sequence");
        Matrix<T> x = embed.forward(tokens);           // (d_model x L)
        x = pe.forward(x);
        for (auto& blk : blocks) x = blk.forward(x);
        x = final_ln.forward(x);
        return head.forward(x);
    }

    // dProbs: gradient w.r.t. the output of the last forward.
    void backward(const Matrix<T>& dProbs) {
        backward_logits(softmax.backward(dProbs));
    }

    // dLogits must correspond to the latest forward/forward_logits call.
    void backward_logits(const Matrix<T>& dLogits) {
        Matrix<T> dX = head.backward(dLogits);
        dX = final_ln.backward(dX);
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
            dX = it->backward(dX);
        dX = pe.backward(dX);
        embed.backward(dX);
    }

    void update(Optimizer<T>& opt) {
        embed.update(opt);
        for (auto& blk : blocks) blk.update(opt);
        final_ln.update(opt);
        head.update(opt);
    }

    void set_training(bool t) {
        for (auto& blk : blocks) blk.set_training(t);
        final_ln.set_training(t);
        head.set_training(t);
    }

    // Parameters are written in the order the layers are called.  The same
    // architecture must be constructed before loading.
    void save(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("TokenTransformer: cannot create " + path);
        detail::write_checkpoint_header<T>(out, "WTTF",
            static_cast<std::uint32_t>(blocks.size()), configuration());
        embed.save_params(out);
        for (auto& blk : blocks) blk.save_params(out);
        final_ln.save_params(out);
        head.save_params(out);
        out.flush();
        if (!out) throw std::runtime_error("TokenTransformer: write failed: " + path);
    }
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("TokenTransformer: cannot open " + path);
        char magic[4] = {};
        in.read(magic, 4);
        if (!in) throw std::runtime_error("TokenTransformer: truncated header");
        in.seekg(0);
        if (std::string(magic, 4) == "WTTF")
            detail::read_checkpoint_header<T>(in, "WTTF",
                static_cast<std::uint32_t>(blocks.size()), configuration(), false);
        // Older transformer files had no header. Component shape checks
        // still validate their payload, and the final read must succeed.
        embed.load_params(in);
        for (auto& blk : blocks) blk.load_params(in);
        final_ln.load_params(in);
        head.load_params(in);
        if (!in) throw std::runtime_error("TokenTransformer: read failed: " + path);
    }

    std::size_t vocab()   const { return vocab_; }
    std::size_t d_model() const { return d_model_; }
    std::size_t max_seq() const { return max_seq_; }

    // Number of trainable parameters.
    std::size_t parameter_count() const {
        const std::size_t d = d_model_;
        std::size_t n = d * vocab_ + (d * vocab_ + vocab_) + 2 * d;   // embed, head, final LN
        for (const auto& blk : blocks) {
            const std::size_t ff = blk.d_ff();
            n += 4 * (d * d + d)          // Wq, Wk, Wv, Wo
               + 2 * 2 * d                // two LayerNorms
               + (ff * d + ff) + (d * ff + d);
        }
        return n;
    }

private:
    std::string configuration() const {
        return "TokenTransformer(" + std::to_string(vocab_) + "," +
            std::to_string(d_model_) + "," + std::to_string(n_heads_) + "," +
            std::to_string(blocks.size()) + "," + std::to_string(d_ff_) + "," +
            std::to_string(max_seq_) + ")";
    }
    std::size_t vocab_, d_model_, max_seq_;
    std::size_t n_heads_, d_ff_;

    Embedding<T>                    embed;
    SinusoidalPositionalEncoding<T> pe;
    std::vector<TransformerEncoderBlock<T>> blocks;
    LayerNorm<T>                    final_ln;
    Dense<T>                        head;
    Softmax<T>                      softmax;
};

} // namespace weft
