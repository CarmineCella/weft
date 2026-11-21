// transformer_chorales / transformer_chorales.cpp
//
// Train an autoregressive transformer on a corpus of four-voice MIDI
// chorales, then sample new chorales from the model.  This is the
// payoff for the attention machinery built in chapters 25-26: a generator
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
//   transformer_chorales generate <model_prefix> <out.mid> [n_steps] [temperature] [seed]
//   transformer_chorales markov   <data_dir> <out.mid> [n_steps] [seed]
//       the per-voice Markov baseline, for comparison by ear
//   transformer_chorales stats    <label> <file.mid | dir> [...]
//       consonance, voice crossing, parallel fifths and note length
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
#include "TokenTransformer.h"
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
constexpr std::size_t GEN_CONTEXT = 128;    // tokens of context when sampling;
                                            // must not exceed the training block

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

    // Per-voice timeline: voice[v][t] = sounding pitch (or -1 = rest), and
    // onset[v][t] = true where a note BEGINS.  The onset flag is what
    // distinguishes one long note from the same pitch struck twice; without
    // it every repeated note would be merged into its predecessor.
    std::vector<std::vector<int>>  voice(N_VOICES, std::vector<int>(T, -1));
    std::vector<std::vector<char>> onset(N_VOICES, std::vector<char>(T, 0));
    for (std::size_t v = 0; v < N_VOICES; ++v) {
        for (const auto& n : f.tracks[v]) {
            const int s = (n.start_tick + hop / 2) / hop;
            const int e = std::min<int>(T, std::max(s + 1,
                              (n.start_tick + n.duration_tick + hop / 2) / hop));
            for (int t = s; t < e; ++t)
                voice[v][t] = n.pitch;
            if (s < T) onset[v][s] = 1;
        }
    }

    // Emit tokens.  REST if nothing sounds; the pitch number where a note
    // begins; HOLD where the note that began earlier is still sounding.
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(T) * N_VOICES + 2);
    out.push_back(TOK_BOS);
    for (int t = 0; t < T; ++t) {
        for (std::size_t v = 0; v < N_VOICES; ++v) {
            const int p = voice[v][t];
            if (p < 0) {
                out.push_back(TOK_REST);
            } else if (t > 0 && !onset[v][t] && voice[v][t - 1] == p) {
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

// The model itself is TokenTransformer (src/TokenTransformer.h): token
// embedding, positional encoding, N_LAYERS causal blocks, a final
// LayerNorm and a Dense layer back to the vocabulary.  This struct only
// fixes its sizes for the chorale task and keeps the .clm file extension.
struct ChoraleLM : TokenTransformer<float> {
    ChoraleLM() : TokenTransformer<float>(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                                       D_FF, MAX_SEQ) {}
    void save(const std::string& prefix) { TokenTransformer<float>::save(prefix + ".clm"); }
    void load(const std::string& prefix) { TokenTransformer<float>::load(prefix + ".clm"); }
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
sample_sequence(ChoraleLM& model, std::size_t n_time_steps,
                float temperature, unsigned seed)
{
    model.set_training(false);
    std::mt19937 rng(seed);
    std::vector<int> ctx = {TOK_BOS};
    const std::size_t target_len = 1 + n_time_steps * N_VOICES;

    while (ctx.size() < target_len) {
        // The model may only be shown what it was trained on: windows of at
        // most GEN_CONTEXT tokens, starting at position 0 of the positional
        // encoding.  So the context is always the LAST GEN_CONTEXT tokens.
        //
        // An earlier version passed the whole sequence so far, up to
        // MAX_SEQ = 1024 tokens.  Training never used positions beyond the
        // block size, so everything after the first 32 time steps was
        // produced from positional encodings the model had never seen --
        // and it showed: voices crossed four times out of five.  The loss
        // curve gave no hint of it, because the loss is measured on
        // training-sized windows.
        // The window also starts at a multiple of 4, as in training, so
        // that position mod 4 still identifies the voice.
        std::size_t start = 0;
        if (ctx.size() > GEN_CONTEXT)
            start = ((ctx.size() - GEN_CONTEXT + N_VOICES - 1) / N_VOICES) * N_VOICES;
        std::vector<int> window(ctx.begin() + static_cast<long>(start), ctx.end());
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
    // Every 10th chorale is held out: never trained on, used to measure
    // how well the model predicts music it has not seen.
    std::vector<std::vector<int>> held_out;
    {
        std::vector<std::vector<int>> train;
        for (std::size_t i = 0; i < corpus.size(); ++i)
            (i % 10 == 5 ? held_out : train).push_back(std::move(corpus[i]));
        corpus = std::move(train);
    }
    std::cout << "split: " << corpus.size() << " training chorales, "
              << held_out.size() << " held out\n";

    // ---- two baselines to compare the transformer against -------------
    // Both are fitted on the training chorales and scored, like the
    // transformer, by the average cross-entropy (in nats per token) of
    // the held-out chorales.
    //   unigram : P(token), ignoring all context.
    //   Markov  : P(token | the same voice's token one time step earlier),
    //             a first-order Markov chain with one transition table per
    //             voice.  In the flat S A T B stream the previous token of
    //             the same voice is the one 4 positions back.  This is the
    //             model the `markov` command samples from.
    {
        const std::size_t V = VOCAB_SIZE;
        std::vector<double> uni(V, 1.0);                             // add-one smoothing
        double uni_total = double(V);
        std::vector<std::vector<double>> big(N_VOICES, std::vector<double>(V * V, 1.0));
        std::vector<std::vector<double>> row_total(N_VOICES, std::vector<double>(V, double(V)));
        for (const auto& seq : corpus)
            for (std::size_t t = 1; t + 1 < seq.size(); ++t) {
                uni[seq[t]] += 1.0; uni_total += 1.0;
                const std::size_t v = (t - 1) % N_VOICES;
                const int prev = (t >= 5) ? seq[t - 4] : TOK_BOS;
                big[v][prev * V + seq[t]] += 1.0;
                row_total[v][prev] += 1.0;
            }
        double h_uni = 0.0, h_big = 0.0; std::size_t n = 0;
        for (const auto& seq : held_out)
            for (std::size_t t = 1; t + 1 < seq.size(); ++t) {
                h_uni -= std::log(uni[seq[t]] / uni_total);
                const std::size_t v = (t - 1) % N_VOICES;
                const int prev = (t >= 5) ? seq[t - 4] : TOK_BOS;
                h_big -= std::log(big[v][prev * V + seq[t]] / row_total[v][prev]);
                ++n;
            }
        std::cout << std::fixed << std::setprecision(4)
                  << "baselines on the held-out chorales (nats per token):\n"
                  << "  uniform over the vocabulary : " << std::log(double(VOCAB_SIZE)) << "\n"
                  << "  unigram                     : " << h_uni / double(n) << "\n"
                  << "  per-voice first-order Markov: " << h_big / double(n) << "\n\n";
    }

    if (corpus.empty()) {
        std::cerr << "no usable corpus -- chorale MIDIs must have 4 tracks "
                     "and produce at least " << (BLOCK + 1) << " tokens\n";
        return 1;
    }

    // ---- model + optimiser ----
    ChoraleLM model;
    CrossEntropy<float> ce;
    Adam<float>         opt(3e-4f);     // lower LR; transformers don't like 1e-3

    // ---- training loop (one chunk per step) ----
    std::mt19937 rng(123);
    const std::size_t STEPS_PER_EPOCH = std::min<std::size_t>(2000, total_tokens / BLOCK);
    std::cout << "training (one " << BLOCK << "-token chunk per step, "
              << STEPS_PER_EPOCH << " steps/epoch):\n";
    std::cout << "epoch    train     held-out    time\n";
    std::cout << "-----    ------    --------    -----\n";

    // Held-out loss: every non-overlapping BLOCK-token chunk of every
    // held-out chorale, no parameter update.
    auto held_out_loss = [&]() {
        double sum = 0.0; std::size_t count = 0;
        for (const auto& seq : held_out)
            for (std::size_t off = 0; off + BLOCK + 1 <= seq.size();
                 off += (BLOCK / N_VOICES) * N_VOICES) {
                std::vector<int> input (seq.begin() + off, seq.begin() + off + BLOCK);
                std::vector<int> target(seq.begin() + off + 1, seq.begin() + off + BLOCK + 1);
                sum += ce.forward(model.forward(input), one_hot_seq(target));
                ++count;
            }
        return count ? sum / double(count) : 0.0;
    };

    auto t_total = clock::now();
    for (int ep = 1; ep <= epochs; ++ep) {
        double sum_loss = 0.0;

        for (std::size_t step = 0; step < STEPS_PER_EPOCH; ++step) {
            // Pick a random chorale, then a random chunk within it.
            std::uniform_int_distribution<std::size_t> pick_file(0, corpus.size() - 1);
            const std::size_t fi = pick_file(rng);
            const std::vector<int>& seq = corpus[fi];
            // The chunk starts at a token index that is a multiple of 4.
            // Index 0 is BOS and indices 1..4 are S, A, T, B of the first
            // time step, so in EVERY chunk positions 1, 2, 3, 0 (mod 4) hold
            // S, A, T, B.  The positional encoding then tells the model
            // which voice it is writing.  With arbitrary offsets it would
            // have to guess that from the notes themselves.
            //
            // One chunk in ten starts at index 0, so that the model sees
            // often enough how a chorale begins; a uniformly random offset
            // would show it an opening about once in two hundred chunks.
            const std::size_t max_k = (seq.size() - BLOCK - 1) / N_VOICES;
            std::size_t off = 0;
            if (std::uniform_int_distribution<int>(0, 9)(rng) != 0)
                off = N_VOICES *
                      std::uniform_int_distribution<std::size_t>(0, max_k)(rng);

            std::vector<int> input (seq.begin() + off, seq.begin() + off + BLOCK);
            std::vector<int> target(seq.begin() + off + 1, seq.begin() + off + BLOCK + 1);

            Matrix<float> probs = model.forward(input);
            Matrix<float> Y     = one_hot_seq(target);
            const float L = ce.forward(probs, Y);
            sum_loss += static_cast<double>(L);

            Matrix<float> dP = ce.backward();
            model.backward(dP);
            model.update(opt);

        }

        {
            const double secs =
                std::chrono::duration<double>(clock::now() - t_total).count();
            std::cout << std::setw(5) << ep << "    "
                      << std::fixed << std::setprecision(4)
                      << sum_loss / double(STEPS_PER_EPOCH) << "    "
                      << held_out_loss() << "      "
                      << std::setprecision(0) << secs << "s\n" << std::flush;
        }

        // End-of-epoch: save and sample.
        model.save(prefix);
        if (ep % 10 != 0 && ep != epochs) continue;      // sample every 10th epoch

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
                     "<model_prefix> <out.mid> [n_steps] [temperature] [seed]\n";
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

    ChoraleLM model;
    try { model.load(prefix); }
    catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n"; return 1;
    }
    auto toks = sample_sequence(model, n_steps, temp, seed);
    midi::write(out, tokens_to_midi(toks));
    std::cout << "wrote " << out << " (" << toks.size() << " tokens)\n";
    return 0;
}

// A chorale from the baseline the transformer is compared against: a
// first-order Markov chain per voice.  Each voice's next token depends
// only on that voice's own previous token; the voices never look at one
// another.  Listening to this next to the transformer's output is the
// audible version of the loss comparison printed by `train`.
static int cmd_markov(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: transformer_chorales markov <data_dir> <out.mid> "
                     "[n_steps] [seed]\n";
        return 1;
    }
    const std::string data_dir = argv[2];
    const std::string out      = argv[3];
    const int n_steps = (argc > 4) ? std::max(1, std::atoi(argv[4])) : 256;
    const unsigned seed = (argc > 5) ? static_cast<unsigned>(std::atoi(argv[5])) : 1u;

    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(data_dir))
        if (entry.path().extension() == ".mid") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());

    // counts[v][prev][next]: one transition table per voice.
    std::vector<std::vector<double>> counts(
        N_VOICES, std::vector<double>(VOCAB_SIZE * VOCAB_SIZE, 0.0));
    std::vector<std::array<int, N_VOICES>> openings;      // first time step of each chorale
    for (const auto& p : paths) {
        std::vector<int> toks;
        try { toks = midi_to_tokens(midi::read(p.string())); }
        catch (const std::exception&) { continue; }
        if (toks.size() < 2 + 2 * N_VOICES) continue;
        std::array<int, N_VOICES> first{};
        for (std::size_t v = 0; v < N_VOICES; ++v) first[v] = toks[1 + v];
        openings.push_back(first);
        // toks[0] is BOS; token 1 + 4*t + v is voice v at time step t.
        for (std::size_t i = 1 + N_VOICES; i + 1 < toks.size(); ++i) {
            const std::size_t v = (i - 1) % N_VOICES;
            counts[v][toks[i - N_VOICES] * VOCAB_SIZE + toks[i]] += 1.0;
        }
    }
    if (openings.empty()) { std::cerr << "no usable MIDI files in " << data_dir << "\n"; return 1; }

    std::mt19937 rng(seed);
    std::array<int, N_VOICES> cur =
        openings[std::uniform_int_distribution<std::size_t>(0, openings.size() - 1)(rng)];
    std::vector<int> toks = {TOK_BOS};
    for (std::size_t v = 0; v < N_VOICES; ++v) toks.push_back(cur[v]);
    for (int t = 1; t < n_steps; ++t)
        for (std::size_t v = 0; v < N_VOICES; ++v) {
            const double* row = &counts[v][cur[v] * VOCAB_SIZE];
            double total = 0.0;
            for (int k = 0; k < VOCAB_SIZE; ++k) total += row[k];
            int next = TOK_HOLD;
            if (total > 0.0) {
                double r = std::uniform_real_distribution<double>(0.0, total)(rng);
                for (int k = 0; k < VOCAB_SIZE; ++k) {
                    r -= row[k];
                    if (r <= 0.0) { next = k; break; }
                }
            }
            if (next == TOK_EOS || next == TOK_BOS) next = TOK_HOLD;
            cur[v] = next;
            toks.push_back(next);
        }
    toks.push_back(TOK_EOS);
    midi::write(out, tokens_to_midi(toks));
    std::cout << "wrote " << out << " (per-voice Markov chain, " << n_steps
              << " steps, from " << openings.size() << " chorales)\n";
    return 0;
}

// Numbers that describe a four-voice chorale, so that generated chorales
// can be compared with Bach's and with the Markov baseline's without
// listening.  All are computed on the 16th-note grid, over the time steps
// at which all four voices sound.
//
//   consonant : how often an upper voice forms a unison, third, fifth,
//               sixth or octave with the bass (intervals 0, 3, 4, 7, 8, 9
//               semitones modulo 12)
//   crossed   : how often two neighbouring voices are in the wrong order
//               (S below A, A below T, or T below B)
//   parallel  : of the moves in which two voices both change pitch, how
//               many go from a perfect fifth or octave to the same interval
//   attack    : average number of quarter notes between attacks in a voice
static int cmd_stats(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: transformer_chorales stats <label> <file.mid | dir> [...]\n";
        return 1;
    }
    const std::string label = argv[2];
    std::vector<fs::path> paths;
    for (int i = 3; i < argc; ++i) {
        const fs::path p = argv[i];
        if (fs::is_directory(p)) {
            for (const auto& e : fs::recursive_directory_iterator(p))
                if (e.path().extension() == ".mid") paths.push_back(e.path());
        } else {
            paths.push_back(p);
        }
    }
    std::sort(paths.begin(), paths.end());

    double cons = 0, cross = 0, par = 0, dur = 0;
    std::size_t files = 0;
    for (const auto& path : paths) {
        midi::File f;
        try { f = midi::read(path.string()); } catch (const std::exception&) { continue; }
        if (f.tracks.size() < N_VOICES) continue;
        const int hop = std::max(1, f.ticks_per_quarter / 4);
        int max_end = 0;
        for (std::size_t v = 0; v < N_VOICES; ++v)
            for (const auto& n : f.tracks[v])
                max_end = std::max(max_end, n.start_tick + n.duration_tick);
        const int T = (max_end + hop - 1) / hop;
        if (T < 8) continue;
        std::vector<std::vector<int>> g(N_VOICES, std::vector<int>(T, -1));
        std::size_t attacks = 0;
        for (std::size_t v = 0; v < N_VOICES; ++v)
            for (const auto& n : f.tracks[v]) {
                const int s = (n.start_tick + hop / 2) / hop;
                const int e = std::min<int>(T, std::max(s + 1,
                                  (n.start_tick + n.duration_tick + hop / 2) / hop));
                for (int t = s; t < e; ++t) g[v][t] = n.pitch;
                ++attacks;
            }
        // Time steps at which all four voices sound.
        std::vector<int> full;
        for (int t = 0; t < T; ++t) {
            bool all = true;
            for (std::size_t v = 0; v < N_VOICES; ++v) if (g[v][t] < 0) all = false;
            if (all) full.push_back(t);
        }
        if (full.empty()) continue;

        std::size_t n_cons = 0, n_cross = 0;
        for (int t : full) {
            for (std::size_t v = 0; v < 3; ++v) {
                const int iv = ((g[v][t] - g[3][t]) % 12 + 12) % 12;
                if (iv == 0 || iv == 3 || iv == 4 || iv == 7 || iv == 8 || iv == 9) ++n_cons;
            }
            if (g[0][t] < g[1][t] || g[1][t] < g[2][t] || g[2][t] < g[3][t]) ++n_cross;
        }
        std::size_t moves = 0, parallels = 0;
        for (std::size_t k = 1; k < full.size(); ++k) {
            const int t0 = full[k - 1], t1 = full[k];
            for (std::size_t a = 0; a < N_VOICES; ++a)
                for (std::size_t b = a + 1; b < N_VOICES; ++b) {
                    if (g[a][t0] == g[a][t1] || g[b][t0] == g[b][t1]) continue;
                    ++moves;
                    const int i0 = ((g[a][t0] - g[b][t0]) % 12 + 12) % 12;
                    const int i1 = ((g[a][t1] - g[b][t1]) % 12 + 12) % 12;
                    if (i0 == i1 && (i1 == 0 || i1 == 7)) ++parallels;
                }
        }
        cons  += double(n_cons) / double(3 * full.size());
        cross += double(n_cross) / double(full.size());
        par   += moves ? double(parallels) / double(moves) : 0.0;
        dur   += double(T) * N_VOICES / double(std::max<std::size_t>(1, attacks)) / 4.0;
        ++files;
    }
    if (files == 0) { std::cerr << "no usable MIDI files\n"; return 1; }
    std::cout << std::left << std::setw(14) << label << std::right << " n=" << std::setw(3) << files
              << std::fixed << std::setprecision(1)
              << "   consonant " << std::setw(5) << 100.0 * cons / files << "%"
              << "   crossed " << std::setw(5) << 100.0 * cross / files << "%"
              << "   parallel 5ths/8ves " << std::setw(4) << 100.0 * par / files << "%"
              << "   quarter notes per attack " << std::setprecision(2) << dur / files << "\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr <<
            "usage:\n"
            "  transformer_chorales train    <data_dir> <model_prefix> "
            "[epochs] [block]\n"
            "  transformer_chorales generate <model_prefix> <out.mid> "
            "[n_steps] [temperature] [seed]\n"
            "  transformer_chorales markov   <data_dir> <out.mid> "
            "[n_steps] [seed]\n"
            "  transformer_chorales stats    <label> <file.mid | dir> [...]\n";
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "train")    return cmd_train   (argc, argv);
    if (cmd == "generate") return cmd_generate(argc, argv);
    if (cmd == "markov")   return cmd_markov  (argc, argv);
    if (cmd == "stats")    return cmd_stats   (argc, argv);
    std::cerr << "unknown command: " << cmd
              << " (try 'train', 'generate', 'markov' or 'stats')\n";
    return 1;
}
