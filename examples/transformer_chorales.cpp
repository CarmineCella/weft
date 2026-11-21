// transformer_chorales / transformer_chorales.cpp
//
// Train an autoregressive transformer on a corpus of four-voice MIDI
// chorales, then sample new chorales from the model.  This is the
// payoff for the attention machinery built in notes 21-22: a generator
// that does something a Markov chain genuinely cannot, because the
// constraint we're modelling -- correct voice leading across four
// simultaneous voices -- is joint over the whole vocal texture, not
// merely sequential within one voice.
//
// What it does, end to end:
//
//   1. Read a directory of MIDI files, each Type 1 with 4 tracks (S, A,
//      T, B in that order).
//   2. Tokenise each file by quantising to 16th notes and emitting one
//      token per voice per time step (so each time step contributes 4
//      tokens in fixed S, A, T, B order).
//   3. Train a small causal-masked transformer on the resulting token
//      stream.  Loss is cross-entropy on next-token prediction.
//   4. Sample new sequences: prime with BOS, ask the model for the
//      next-token distribution at the last position, sample, append,
//      repeat.  Convert the resulting tokens back to MIDI and write
//      a four-track .mid file.
//
// Vocabulary (134 tokens):
//   0..127   MIDI pitch attack  (a new note at this pitch starts here)
//   128      REST              (this voice is silent at this time step)
//   129      HOLD              (the same pitch from the previous step
//                               continues -- used for notes longer than
//                               one 16th note)
//   130      BOS               (begin-of-sequence sentinel)
//   131      EOS               (end-of-sequence sentinel)
//   132      PAD               (unused at the moment; reserved)
//   133      UNK               (any token we couldn't map; should not
//                               appear in well-formed training data)
//
// Usage:
//   transformer_chorales train    <data_dir> <model_prefix> [epochs] [block]
//   transformer_chorales generate <model_prefix> <out.mid> [n_steps] [temperature]
//
// `epochs` default 5, `block` default 128 tokens (32 time-steps),
// `n_steps` default 256 time-steps (~16 bars at 4/4), `temperature`
// default 1.0 (lower = more conservative, higher = more random).

#include "Adam.h"
#include "CrossEntropy.h"
#include "Dense.h"
#include "Embedding.h"
#include "LayerNorm.h"
#include "Matrix.h"
#include "Midi.h"
#include "MultiHeadAttention.h"
#include "ReLU.h"
#include "ScaledDotProductAttention.h"
#include "SinusoidalPositionalEncoding.h"
#include "Softmax.h"
#include "TransformerEncoderBlock.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace weft;
namespace fs = std::filesystem;

// ---- vocabulary ---------------------------------------------------------
constexpr int VOCAB_SIZE = 134;
constexpr int TOK_REST   = 128;
constexpr int TOK_HOLD   = 129;
constexpr int TOK_BOS    = 130;
constexpr int TOK_EOS    = 131;
// Token 132 = PAD, 133 = UNK; reserved but unused in v1.

// ---- model size ---------------------------------------------------------
constexpr std::size_t D_MODEL    = 128;
constexpr std::size_t N_HEADS    = 4;
constexpr std::size_t D_FF       = 4 * D_MODEL;
constexpr std::size_t N_LAYERS   = 2;
constexpr std::size_t MAX_SEQ    = 1024;     // positional encoding budget
constexpr std::size_t N_VOICES   = 4;

// =========================================================================
// Tokenisation
// =========================================================================

// Convert one MIDI file (4 tracks expected) to a flat token stream.
// Quantises start times and durations to the nearest 16th note.
// Returns BOS-prefixed, EOS-terminated tokens in S-A-T-B order per
// time step.
static std::vector<int> midi_to_tokens(const midi::File& f) {
    if (f.tracks.size() < N_VOICES) return {};
    const int hop = std::max(1, f.ticks_per_quarter / 4);   // 16th note in ticks

    // For each voice, build a length-T array of (active_pitch or -1).
    // First pass: figure out total length T as the latest end across voices.
    int max_end = 0;
    for (std::size_t v = 0; v < N_VOICES; ++v)
        for (const auto& n : f.tracks[v])
            max_end = std::max(max_end, n.start_tick + n.duration_tick);
    const int T = (max_end + hop - 1) / hop;
    if (T <= 0) return {};

    // Per-voice timeline: voice[v][t] = pitch (or -1 = rest)
    std::vector<std::vector<int>> voice(N_VOICES, std::vector<int>(T, -1));
    for (std::size_t v = 0; v < N_VOICES; ++v) {
        for (const auto& n : f.tracks[v]) {
            const int s = (n.start_tick + hop / 2) / hop;
            const int e = std::min<int>(T, std::max(s + 1,
                              (n.start_tick + n.duration_tick + hop / 2) / hop));
            for (int t = s; t < e; ++t)
                voice[v][t] = n.pitch;
        }
    }

    // Emit tokens.  REST if -1; HOLD if same as previous step; else the
    // pitch number as an attack.
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(T) * N_VOICES + 2);
    out.push_back(TOK_BOS);
    for (int t = 0; t < T; ++t) {
        for (std::size_t v = 0; v < N_VOICES; ++v) {
            const int p = voice[v][t];
            if (p < 0) {
                out.push_back(TOK_REST);
            } else if (t > 0 && voice[v][t - 1] == p) {
                out.push_back(TOK_HOLD);
            } else {
                out.push_back(p);                            // attack at pitch p
            }
        }
    }
    out.push_back(TOK_EOS);
    return out;
}

// Inverse: read the token stream back into a 4-track MIDI file.
// Skips BOS/EOS and any unknown tokens.  Notes are accumulated per
// voice; a new attack closes the previous note in that voice.
static midi::File tokens_to_midi(const std::vector<int>& toks,
                                 int ticks_per_quarter = 480) {
    midi::File f;
    f.ticks_per_quarter = ticks_per_quarter;
    f.tracks.assign(N_VOICES, midi::Track{});
    const int hop = ticks_per_quarter / 4;

    // Per-voice rolling state: which pitch is currently sounding and when
    // did it start.
    std::array<int, N_VOICES> active_pitch{};
    std::array<int, N_VOICES> active_start{};
    for (std::size_t v = 0; v < N_VOICES; ++v) {
        active_pitch[v] = -1;
        active_start[v] = 0;
    }

    int t = 0;
    std::size_t v_idx = 0;        // which voice the next token applies to (0..3, cycles)

    auto close_voice = [&](std::size_t v, int end_tick) {
        if (active_pitch[v] >= 0 && end_tick > active_start[v]) {
            f.tracks[v].push_back({active_pitch[v], 96,
                                   active_start[v],
                                   end_tick - active_start[v]});
        }
        active_pitch[v] = -1;
    };

    for (int tok : toks) {
        if (tok == TOK_BOS || tok == TOK_EOS) continue;
        if (tok < 0 || tok >= VOCAB_SIZE) { tok = TOK_REST; }

        const int now = t * hop;
        if (tok == TOK_REST) {
            close_voice(v_idx, now);
        } else if (tok == TOK_HOLD) {
            // Nothing to do: the active note keeps sounding.
            // If there's no active note (model error), we silently
            // continue treating the voice as resting.
        } else if (tok >= 0 && tok < 128) {
            // Attack at this pitch: close any previous note then start fresh.
            close_voice(v_idx, now);
            active_pitch[v_idx] = tok;
            active_start[v_idx] = now;
        }

        ++v_idx;
        if (v_idx == N_VOICES) { v_idx = 0; ++t; }
    }
    // Flush any notes that were still active at the end.
    const int end_tick = t * hop;
    for (std::size_t v = 0; v < N_VOICES; ++v) close_voice(v, end_tick);
    return f;
}

// =========================================================================
// Model
// =========================================================================

struct CharLM {
    Embedding<float>                    embed;
    SinusoidalPositionalEncoding<float> pe;
    std::vector<TransformerEncoderBlock<float>> blocks;
    LayerNorm<float>                    final_ln;
    Dense<float>                        head;
    Softmax<float>                      softmax;

    CharLM()
        : embed(VOCAB_SIZE, D_MODEL),
          pe(D_MODEL, MAX_SEQ),
          final_ln(D_MODEL),
          head(D_MODEL, VOCAB_SIZE)
    {
        blocks.reserve(N_LAYERS);
        for (std::size_t i = 0; i < N_LAYERS; ++i) {
            blocks.emplace_back(D_MODEL, N_HEADS, D_FF);
            blocks.back().set_causal(true);     // autoregressive
        }
    }

    // Forward returns Softmax probabilities over the vocabulary, shape
    // (VOCAB_SIZE x seq_len).  Caches intermediates for backward.
    Matrix<float> forward(const std::vector<int>& tokens) {
        Matrix<float> x = embed.forward(tokens);   // (d_model x L)
        x = pe.forward(x);
        for (auto& blk : blocks) x = blk.forward(x);
        x = final_ln.forward(x);
        Matrix<float> logits = head.forward(x);    // (vocab x L)
        return softmax.forward(logits);
    }

    void backward(const Matrix<float>& dProbs) {
        Matrix<float> dLogits = softmax.backward(dProbs);
        Matrix<float> dX      = head.backward(dLogits);
        dX = final_ln.backward(dX);
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
            dX = it->backward(dX);
        dX = pe.backward(dX);
        embed.backward(dX);
    }

    void update(Optimizer<float>& opt) {
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

    void save(const std::string& prefix) {
        std::ofstream out(prefix + ".clm", std::ios::binary);
        if (!out) throw std::runtime_error("cannot create " + prefix + ".clm");
        embed.save_params(out);
        for (auto& blk : blocks) blk.save_params(out);
        final_ln.save_params(out);
        head.save_params(out);
    }
    void load(const std::string& prefix) {
        std::ifstream in(prefix + ".clm", std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + prefix + ".clm");
        embed.load_params(in);
        for (auto& blk : blocks) blk.load_params(in);
        final_ln.load_params(in);
        head.load_params(in);
    }
};

// =========================================================================
// Training and sampling
// =========================================================================

// Build a one-hot target matrix (vocab x len) from a vector of token ids.
static Matrix<float> one_hot_seq(const std::vector<int>& ids) {
    Matrix<float> Y(VOCAB_SIZE, ids.size(), 0.0f);
    for (std::size_t t = 0; t < ids.size(); ++t)
        Y(static_cast<std::size_t>(ids[t]), t) = 1.0f;
    return Y;
}

// Sample one token from the probability distribution at column `col` of
// `probs`, optionally rescaled by temperature.  Lower T = sharper (more
// confident, less varied); higher T = flatter (more random).
static int sample_token(const Matrix<float>& probs, std::size_t col,
                        float temperature, std::mt19937& rng) {
    // Re-temperature: p' ~ p^(1/T), then renormalise.
    std::vector<float> p(VOCAB_SIZE);
    float sum = 0.0f;
    const float inv_T = 1.0f / std::max(1e-6f, temperature);
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        // probs is already softmax output; raise to 1/T in log-space to
        // avoid overflow for tiny probabilities.
        const float lp = std::log(std::max(1e-12f, probs(v, col))) * inv_T;
        p[v] = std::exp(lp);
        sum += p[v];
    }
    for (int v = 0; v < VOCAB_SIZE; ++v) p[v] /= sum;
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const float r = U(rng);
    float cum = 0.0f;
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        cum += p[v];
        if (r <= cum) return v;
    }
    return VOCAB_SIZE - 1;
}

// Generate a sequence of `n_time_steps` 16th-note steps (so 4 *
// n_time_steps voice tokens).  Returns the full token stream including
// the priming BOS.
static std::vector<int>
sample_sequence(CharLM& model, std::size_t n_time_steps,
                float temperature, unsigned seed)
{
    model.set_training(false);
    std::mt19937 rng(seed);
    std::vector<int> ctx = {TOK_BOS};
    const std::size_t target_len = 1 + n_time_steps * N_VOICES;

    while (ctx.size() < target_len && ctx.size() < MAX_SEQ) {
        // Window the context to the last MAX_SEQ - 1 tokens if it grows
        // past the positional encoding budget (won't happen with default
        // MAX_SEQ = 1024 unless the user asks for very long output).
        std::vector<int> window = ctx;
        if (window.size() >= MAX_SEQ) {
            window.erase(window.begin(),
                         window.begin() + (window.size() - MAX_SEQ + 1));
        }
        Matrix<float> probs = model.forward(window);
        const int next = sample_token(probs, probs.cols() - 1, temperature, rng);
        ctx.push_back(next);
        if (next == TOK_EOS) break;
    }
    model.set_training(true);
    return ctx;
}

// =========================================================================
// Commands
// =========================================================================

static int cmd_train(int argc, char** argv) {
    using clock = std::chrono::steady_clock;
    if (argc < 4) {
        std::cerr << "usage: transformer_chorales train <data_dir> "
                     "<model_prefix> [epochs] [block]\n";
        return 1;
    }
    const std::string data_dir = argv[2];
    const std::string prefix   = argv[3];
    const int         epochs   = (argc > 4) ? std::atoi(argv[4]) : 5;
    const std::size_t BLOCK    = (argc > 5) ? std::max(8, std::atoi(argv[5])) : 128;

    std::cout << "weft :: transformer_chorales train\n"
              << "  data dir:    " << data_dir << "\n"
              << "  prefix:      " << prefix   << "\n"
              << "  epochs:      " << epochs   << "\n"
              << "  block size:  " << BLOCK    << "\n"
              << "  d_model:     " << D_MODEL  << "\n"
              << "  n_layers:    " << N_LAYERS << "\n"
              << "  n_heads:     " << N_HEADS  << "\n"
              << "  vocab:       " << VOCAB_SIZE << "\n\n";

    // ---- read corpus and tokenise everything up front ----
    std::cout << "scanning for .mid files..." << std::flush;
    std::vector<fs::path> paths;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(data_dir))
            if (entry.path().extension() == ".mid"
             || entry.path().extension() == ".MID"
             || entry.path().extension() == ".midi")
                paths.push_back(entry.path());
    } catch (const std::exception& e) {
        std::cerr << "\nerror scanning: " << e.what() << "\n"; return 1;
    }
    std::sort(paths.begin(), paths.end());
    std::cout << " " << paths.size() << " files\n";

    std::vector<std::vector<int>> corpus;
    std::size_t total_tokens = 0;
    for (const auto& p : paths) {
        try {
            auto toks = midi_to_tokens(midi::read(p.string()));
            if (toks.size() > BLOCK + 1) {
                corpus.push_back(std::move(toks));
                total_tokens += corpus.back().size();
            }
        } catch (const std::exception& e) {
            std::cerr << "  warning: " << p.filename().string()
                      << ": " << e.what() << "\n";
        }
    }
    std::cout << "tokenised " << corpus.size() << " usable files, "
              << total_tokens << " tokens total ("
              << (corpus.empty() ? 0.0 : double(total_tokens) / corpus.size())
              << " avg per file)\n\n";
    if (corpus.empty()) {
        std::cerr << "no usable corpus -- chorale MIDIs must have 4 tracks "
                     "and produce at least " << (BLOCK + 1) << " tokens\n";
        return 1;
    }

    // ---- model + optimiser ----
    CharLM model;
    CrossEntropy<float> ce;
    Adam<float>         opt(3e-4f);     // lower LR; transformers don't like 1e-3

    // ---- training loop (one chunk per step) ----
    std::mt19937 rng(123);
    const std::size_t STEPS_PER_EPOCH = std::min<std::size_t>(2000, total_tokens / BLOCK);
    std::cout << "training (one " << BLOCK << "-token chunk per step, "
              << STEPS_PER_EPOCH << " steps/epoch):\n";
    std::cout << "epoch    step      loss      perp     time\n";
    std::cout << "-----    -----    ------    ------    -----\n";

    auto t_total = clock::now();
    for (int ep = 1; ep <= epochs; ++ep) {
        auto t_ep = clock::now();
        double sum_loss = 0.0;
        const std::size_t LOG = std::min<std::size_t>(100, std::max<std::size_t>(10, STEPS_PER_EPOCH / 8));

        for (std::size_t step = 0; step < STEPS_PER_EPOCH; ++step) {
            // Pick a random chorale, then a random chunk within it.
            std::uniform_int_distribution<std::size_t> pick_file(0, corpus.size() - 1);
            const std::size_t fi = pick_file(rng);
            const std::vector<int>& seq = corpus[fi];
            std::uniform_int_distribution<std::size_t>
                pick_off(0, seq.size() - BLOCK - 1);
            const std::size_t off = pick_off(rng);

            std::vector<int> input (seq.begin() + off, seq.begin() + off + BLOCK);
            std::vector<int> target(seq.begin() + off + 1, seq.begin() + off + BLOCK + 1);

            Matrix<float> probs = model.forward(input);
            Matrix<float> Y     = one_hot_seq(target);
            const float L = ce.forward(probs, Y);
            sum_loss += static_cast<double>(L);

            Matrix<float> dP = ce.backward();
            model.backward(dP);
            model.update(opt);

            if ((step + 1) % LOG == 0) {
                const double avg = sum_loss / LOG;
                const double perp = std::exp(avg);
                const double secs =
                    std::chrono::duration<double>(clock::now() - t_ep).count();
                std::cout << std::setw(5) << ep << "    "
                          << std::setw(5) << (step + 1) << "    "
                          << std::fixed << std::setprecision(4)
                          << std::setw(6) << avg << "    "
                          << std::setprecision(2)
                          << std::setw(6) << perp << "    "
                          << std::setprecision(1)
                          << std::setw(5) << secs << "s\n" << std::flush;
                sum_loss = 0.0;
            }
        }

        // End-of-epoch: save and sample.
        model.save(prefix);
        std::cout << "  [model saved -> " << prefix << ".clm]\n";

        auto sample = sample_sequence(model, /*time_steps=*/32,
                                      /*temperature=*/1.0f,
                                      /*seed=*/static_cast<unsigned>(ep * 17));
        const std::string sample_path =
            prefix + ".epoch" + std::to_string(ep) + ".mid";
        try {
            midi::write(sample_path, tokens_to_midi(sample));
            std::cout << "  [sample -> " << sample_path << "]\n";
        } catch (const std::exception& e) {
            std::cerr << "  sample write failed: " << e.what() << "\n";
        }
    }

    const double total =
        std::chrono::duration<double>(clock::now() - t_total).count();
    std::cout << "\ntotal training time: " << std::fixed << std::setprecision(1)
              << total << "s\n";
    return 0;
}

static int cmd_generate(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: transformer_chorales generate "
                     "<model_prefix> <out.mid> [n_steps] [temperature]\n";
        return 1;
    }
    const std::string prefix = argv[2];
    const std::string out    = argv[3];
    const std::size_t n_steps = (argc > 4) ? std::max(1, std::atoi(argv[4])) : 256;
    const float       temp    = (argc > 5) ? std::atof(argv[5]) : 1.0f;
    const unsigned    seed    = (argc > 6) ? static_cast<unsigned>(std::atoi(argv[6])) : 1u;

    std::cout << "weft :: transformer_chorales generate\n"
              << "  model:        " << prefix << ".clm\n"
              << "  out:          " << out    << "\n"
              << "  n_steps:      " << n_steps << " (16th notes)\n"
              << "  temperature:  " << temp   << "\n"
              << "  seed:         " << seed   << "\n\n";

    CharLM model;
    try { model.load(prefix); }
    catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n"; return 1;
    }
    auto toks = sample_sequence(model, n_steps, temp, seed);
    midi::write(out, tokens_to_midi(toks));
    std::cout << "wrote " << out << " (" << toks.size() << " tokens)\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr <<
            "usage:\n"
            "  transformer_chorales train    <data_dir> <model_prefix> "
            "[epochs] [block]\n"
            "  transformer_chorales generate <model_prefix> <out.mid> "
            "[n_steps] [temperature] [seed]\n";
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "train")    return cmd_train   (argc, argv);
    if (cmd == "generate") return cmd_generate(argc, argv);
    std::cerr << "unknown command: " << cmd
              << " (try 'train' or 'generate')\n";
    return 1;
}
