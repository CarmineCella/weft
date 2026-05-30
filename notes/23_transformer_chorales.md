# 23 — A transformer that writes chorales

The examples in this book so far have done things classical signal
processing already does. Phase vocoders cross-synthesised timbres in
the 1980s. Markov chains have generated plausible-sounding folk
melodies since the 1950s. If a neural-network book stops at examples
that match those old baselines, the reader is entitled to ask why we
bothered with all the gradient bookkeeping.

This note is the answer for the attention chapters. The task is
**generation of four-voice chorales in the style of Bach**, and the
reason it matters is that **Markov chains genuinely cannot do this.**
A first-order Markov over one voice produces a plausible melody.
Extending to four voices means joint transitions over the cartesian
product of voice states, an exponential blow-up that hits the
sparsity wall immediately on any realistic corpus. Higher-order
Markov chains exacerbate the same problem. The constraint we actually
want to model — *correct voice leading across four simultaneous
voices over many bars* — is a long-range, joint, structured
dependency, and that's precisely the regime where transformer
attention exists to live.

This was the framing that motivated DeepBach (Hadjeres, Pachet,
Nielsen 2017) and Coconet (Huang et al. 2017): chorales are the
testbed where neural sequence models obviously beat their classical
predecessors. We're going to build a much smaller version of the
same idea — autoregressive (DeepBach is iteratively-rewriting,
Coconet is masked-orderless; we'll do the simpler one) — but the
core demonstration is the same.


## 1. Three small library additions

The pieces we have from notes 21-22 (`ScaledDotProductAttention`,
`MultiHeadAttention`, `SinusoidalPositionalEncoding`, `LayerNorm`,
`TransformerEncoderBlock`) are an *encoder*. To make them generative
we need three more things.

**Causal masking.** A bidirectional encoder lets every position see
every other position. For autoregressive generation each position
must only see *earlier* positions — otherwise training is meaningless
because the model trivially predicts the next token by looking at the
answer. The fix is one line in `ScaledDotProductAttention::forward`:
before the softmax, set `S[j, i] = -∞` for `j > i`. After softmax
those weights become zero. We expose it as a `causal` flag on
`ScaledDotProductAttention::forward` and a `set_causal(bool)` member
on `MultiHeadAttention` and `TransformerEncoderBlock` so callers can
toggle it per training run without rebuilding the architecture. The
finite-difference test passes at `~1e-12` in double precision with
the mask active, confirming the masked backward is correct.

**Embedding.** Until now every layer took continuous-valued matrices.
A language model takes integer token IDs, so we need a learnable
lookup table. `Embedding<T>` holds a `(d_model × vocab_size)`
parameter matrix `W`; forward returns the column-by-column gather
indexed by the input token list; backward scatters the upstream
gradient back into the columns that were looked up (and *adds*, so a
repeated token gets the sum of its uses — a subtle correctness point
the test verifies explicitly). It's not a `Layer<T>` subclass because
its input is `std::vector<int>` rather than `Matrix<T>`; we just give
it a compatible forward/backward/update interface and call it
directly in the example.

**MIDI I/O.** A binary format with big-endian multi-byte values and
delta-time variable-length quantities. Our `Midi.h` does Format 0
and Format 1 read, Format 1 write, with note-on/note-off events,
tempo, and the "running status" optimisation in the parser. Other
event types (sysex, control changes, pitch bend, program changes) are
*skipped* on read and not emitted on write — a deliberate scope
restriction. About 250 lines, with a round-trip test covering single-
track, four-track, and re-articulated-pitch cases.


## 2. The tokenisation that decides everything

Tokenisation is the most consequential choice and the least
theoretically interesting one. Every successful music transformer
paper has invented its own.

For four-voice chorales the scheme we use is the one DeepBach proves
adequate:

1. **Quantise** every voice to 16th-note onsets. This loses
   ornamentation but keeps the structural content of a chorale.
2. **Flatten** the four voices into one stream, emitting four tokens
   per 16th-note time step in the fixed order S, A, T, B.
3. **At each step in each voice**, emit one of three things: the
   MIDI pitch number (0-127) if a *new note* begins here; `HOLD` if
   the same pitch from the previous step is still sounding; `REST`
   if the voice is silent.

A 32-bar chorale at 4/4 with this scheme is `4 × 16 × 32 + 2 ≈ 2050`
tokens (the +2 are `BOS` and `EOS` sentinels). Total vocabulary is
small: 128 pitches + REST + HOLD + BOS + EOS + 2 reserved = 134
tokens.

Two practical consequences are worth flagging because they shape what
the model learns:

- **Voice order is rigid.** The model always sees `S0 A0 T0 B0 S1 A1
  T1 B1 ...`, so it learns that position-mod-4 means
  voice-in-the-texture. That's helpful — voice ranges are
  consistent, voice leading is predictable within position-mod-4
  groups. It's also a constraint: the model can't choose to be
  monophonic, can't drop a voice, can't add a fifth.

- **HOLD is the workhorse.** Most 16th-note steps don't start a new
  note; they continue a longer one. A model that emits too many
  attacks will produce a 16th-note staccato hailstorm. A model that
  emits too many HOLDs will produce music made of whole notes. The
  emission balance has to come out of the training data, which means
  the training data has to actually sound like chorales (4-voice
  texture, mostly quarter and half notes, occasional 16ths).


## 3. The model

A small decoder-only stack, the standard GPT-style recipe:

```
tokens (L integers)
   |
   v
Embedding(vocab, d_model)          -> (d_model x L)
   |
   v
SinusoidalPositionalEncoding       -> (d_model x L)
   |
   v
TransformerEncoderBlock x N        -> (d_model x L)      (causal mask on)
   |
   v
LayerNorm                          -> (d_model x L)
   |
   v
Dense(d_model -> vocab)            -> (vocab x L)
   |
   v
Softmax                            -> (vocab x L)
```

Our defaults are `d_model = 128`, `n_heads = 4`, `n_layers = 2`,
`d_ff = 4 * d_model`. That's about 440k parameters — small enough to
train on a CPU, large enough to learn non-trivial voice leading.
Block size for training is 128 tokens (32 time steps = 8 bars at
4/4). At inference we step one token at a time.

Per training step:

1. Pick a random chorale token sequence from the corpus.
2. Pick a random offset inside it; take a 128-token window as
   `input` and the same window shifted by one as `target`.
3. Forward through the model with the causal mask on. The output is
   a probability over 134 tokens at every one of the 128 positions
   in the chunk.
4. Loss = cross-entropy between predicted probabilities and
   one-hot targets, averaged across positions.
5. Backward, Adam step.

Because every position in the chunk produces a prediction (not just
the last), one chunk gives us 128 supervised gradients for the cost
of one forward. That's the central efficiency win of training
transformers parallel-in-time. (At inference, we have to go back to
one token at a time — there's no avoiding that for sampling.)


## 4. Sampling

Generation is a loop:

```
context = [BOS]
loop:
    probs = model.forward(context)       # (vocab x len(context))
    next = sample_from(probs[:, -1], temperature)
    context.append(next)
    if next == EOS: break
```

The `temperature` parameter rescales the probability distribution
before sampling: low temperatures (0.5-0.8) bias toward the most
confident predictions, producing more conservative output; high
temperatures (1.0-1.5) flatten the distribution, producing more
varied output that occasionally goes off the rails. Temperature 1.0
samples directly from the model's distribution.

A small adversarial detail: after sampling enough tokens to fill
`MAX_SEQ`, the positional encoding runs out, so we have to *slide*
the context window forward, dropping the oldest tokens. The
generated music's coherence then falls to whatever the model's
practical attention range turned out to be in training. With 2-layer
attention and `block=128` training chunks, the practical range is on
the order of the training block size — beyond that, the model is
generalising past what it saw.


## 5. Honest expectations

For the book's purposes we run this against whatever Bach chorale
corpus you have on disk. Sources commonly cited in the ML music
literature: the `music21` corpus, the JSB Chorales dataset
(Boulanger-Lewandowski et al. 2012, ~400 chorales), or hand-picked
MIDI files from sites like musedata.org. Each chorale needs to be a
4-track MIDI with the four voices as separate tracks in S, A, T, B
order. The example reads any directory of such files.

With a real chorale corpus, what to expect:

- **Loss curve**: starts around `log(134) ≈ 4.9` (uniform-over-vocab
  baseline), drops to about 1.0-1.5 within a few epochs, slowly
  approaches ~0.7 with longer training. Perplexity around 2 means
  the model is effectively choosing among 2 plausible next tokens at
  each step — which lines up with the fact that within a chorale,
  most positions have a small number of musically plausible
  continuations.

- **Generated output (early)**: stays in the right pitch ranges per
  voice, voicings sit in approximately the right vertical alignment,
  but voice leading is loose — parallel fifths, sustained
  dissonances, abrupt range jumps.

- **Generated output (well-trained)**: phrases that hold together,
  cadences that approximately resolve, mostly-correct voice ranges.
  Not Bach, but more musically coherent than what an n-gram on the
  same data would produce. The honest claim is "qualitatively better
  than first-order Markov, not equal to a careful human."

- **What this won't fix on its own**: rhythm beyond 16th-note
  quantisation, dynamics, articulation, lyric setting, large-scale
  form. All of these need either richer tokenisation (REMI-style
  event tokens, encoded velocity), explicit conditioning (give the
  model a key signature, a chorale text, a chord sequence to harmonise),
  or training data with the relevant variation.

These are all the honest places future iterations can go. The library
infrastructure now supports them in principle — embed the right
token type, train longer, use a deeper model.


## 6. What this proves

The point isn't that the model is good. It's that *this kind of model
is possible* — that everything we've built across all the previous
notes composes into a generative system that produces structurally
coherent, joint, multi-voice musical output from a corpus, in a way
that classical methods on the same input couldn't. The dense MLP
classifier, the convnet, the audio VAE, the ConvVAE — none of those
demonstrated something a phase vocoder or Markov chain doesn't
already do. This one does. That's the threshold neural networks
needed to cross for the book's pedagogical argument to work, and
crossing it is what the attention layer actually buys us.

The closing capstone (note 24) brings attention together with
matching pursuit from note 3: a generative orchestration system that
chooses atoms from a learned dictionary, conditioned by attention on
the previous selections. That's where the Part II machinery finally
points at music itself rather than at a benchmark.
