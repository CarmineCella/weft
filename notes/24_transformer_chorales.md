# 24 — A transformer that writes chorales

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

This was the framing that motivated DeepBach [@hadjeres2017deepbach] and Coconet [@huang2017coconet]: chorales are the
testbed where neural sequence models obviously beat their classical
predecessors. We're going to build a much smaller version of the
same idea — autoregressive (DeepBach is iteratively-rewriting,
Coconet is masked-orderless; we'll do the simpler one) — but the
core demonstration is the same.


## 1. What this note adds

Notes 22 and 23 built `MultiHeadAttention`, `SinusoidalPositionalEncoding`,
`LayerNorm` and `TransformerEncoderBlock`. With the causal mask of
note 22 switched on (`set_causal(true)` on every block), a stack of
those blocks is already the engine of a generator: each position sees
only the positions before it, so the model can be trained to predict
the next token and then run one token at a time. Two things are still
missing, and neither is about attention.

**Tokens in.** A language model takes integer token IDs, not vectors.
`Embedding<T>`, the lookup table we met in note 21 with twelve
instruments, does the conversion; here its vocabulary is the 134
tokens defined below. It holds a `(d_model × vocab_size)` parameter
matrix `W`; forward returns, column by column, the vectors of the
input tokens; backward scatters the upstream gradient back into the
columns that were looked up, and *adds*, so that a token occurring
five times in a sequence receives the sum of five gradients. The test
checks that case explicitly, and also that a token which does not
occur receives exactly zero. It's not a `Layer<T>` subclass because
its input is `std::vector<int>` rather than `Matrix<T>`; we give it a
compatible forward/backward/update interface and call it directly in
the example.

**MIDI in and out.** A binary format with big-endian multi-byte values
and delta times stored as variable-length quantities. `Midi.h` reads
Format 0 and Format 1 and writes Format 1, with note-on and note-off
events, tempo, and the "running status" abbreviation in the parser.
Other event types (sysex, control changes, pitch bend, program
changes) are *skipped* on read and not emitted on write, a deliberate
restriction of scope. About 330 lines.

The round-trip test in `tests/test_midi.cpp` found a bug on its first
run, of the kind that never shows up by ear on a chorale. The reader
was taking each note's velocity from its *note-off* event. In MIDI
both events carry a velocity, but only the note-on's is the loudness
of the note; the note-off's is a release speed, and most files simply
write 64 there. Every note therefore came back at velocity 64. A
dataset in which all notes are equally loud hides that completely,
and a test that writes a note at velocity 90 and reads it back does
not.


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
`d_ff = 4 * d_model`. That's about 430k parameters — small enough to
train on a CPU, large enough to learn non-trivial voice leading.
Block size for training is 128 tokens (32 time steps = 8 bars at
4/4). At inference we step one token at a time.

Per training step:

1. Pick a random chorale token sequence from the corpus.
2. Pick an offset inside it that is a multiple of 4; take a 128-token
   window as `input` and the same window shifted by one as `target`.
   One window in ten starts at the very beginning of its chorale.
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

**Why the offset is a multiple of 4.** Token 0 is `BOS` and tokens
1 to 4 are the four voices of the first time step, so a window that
starts at a multiple of 4 always has S, A, T, B at positions 1, 2, 3,
0 (mod 4). The positional encoding then tells the model which voice
it is writing. We did not do this at first. With arbitrary offsets
the model has to work out from the notes themselves which voice comes
next, and section 5 shows what that cost. The one-in-ten rule is
there for a similar reason: a uniformly random offset shows the model
the opening of a chorale about once in two hundred windows, and a
generator that has hardly seen an opening starts badly.


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

**The context must look like training.** The model was trained on
windows of 128 tokens that begin at position 0 of the positional
encoding. At generation time it must be shown the same thing: the
*last* 128 tokens (rounded so that the window still starts at a
multiple of 4), not the whole sequence so far. Coherence beyond 32
time steps therefore comes only from what the window still contains.

This is worth stating because our first sampler got it wrong, and
nothing in training said so. It passed the whole sequence, up to the
1024 positions the encoding table holds. Positions beyond 127 had
never occurred in training, so after the first two bars the model was
working from encodings it had never seen. The loss curve looked
excellent throughout, because the loss is measured on training-sized
windows.


## 5. Results

The corpus is the set of four-part Bach chorales that ships with
music21 [@cuthbert2010music21]; `data/make_chorales.py` writes 365 of
them as four-track MIDI files, each transposed to C major or A minor
so that a small model does not have to learn every key separately.
One chorale in ten is held out.

**Prediction.** The program fits two baselines on the training
chorales and scores them, like the transformer, by the average
cross-entropy of the held-out chorales, in nats per token:

| Model | Held-out loss |
|---|---|
| Uniform over the 134 tokens | 4.90 |
| Unigram | 1.36 |
| First-order Markov chain per voice | 1.28 |
| Transformer, 1 epoch (10 s) | 0.99 |
| Transformer, 10 epochs | 0.60 |
| Transformer, 40 epochs (8 min) | 0.49 |

A loss of 0.49 is a perplexity of 1.6: at each step the model is, on
average, undecided between fewer than two tokens. The transformer
passes the Markov chain during its first epoch.

**Generation.** A low loss says the model predicts Bach well. It does
not say that what it *writes* is any good, and the two can come apart
badly, so we measure the generated music directly. For each chorale
we compute how often the upper voices form a consonance with the bass
(a unison, third, fifth, sixth or octave), how often two voices
cross, how many of the moves between two voices are parallel fifths
or octaves, and the average length of a note. Ten chorales of 256
time steps from each generator, against the 365 originals:

| | Consonant with bass | Voices crossed | Parallel 5ths/8ves | Quarter notes per attack |
|---|---|---|---|---|
| Bach | 85% | 2% | 0.2% | 0.95 |
| Markov chain per voice | 55% | 45% | 3.7% | 1.08 |
| Transformer | 78% | 13% | 5.1% | 1.03 |

The Markov chain produces four plausible lines that ignore one
another: its voices cross almost half the time and its harmony is
not far from what chance would give. The transformer keeps the voices
in order most of the time and is much closer to Bach in its harmony,
with the right rhythmic density. That is the difference attention was
supposed to make, and it is measurable.

It is also not Bach. Voices still cross six times as often as in the
originals, and parallel fifths and octaves, which the chorale style
avoids almost completely, are *more* frequent than in the Markov
output: the model has learned that voices move together, but not the
rule about how they may not.

**Two bugs that the loss did not show.** The table above is the third
version. The first, with the sampler that passed the whole sequence
(section 4), produced chorales whose voices crossed 82% of the time,
worse than the Markov chain. Fixing the sampler brought the harmony
up but left the crossing at 80%. Aligning the training windows to
time steps (section 3) is what brought it to 13%, and it also lowered
the held-out loss from 0.52 to 0.49. In both cases the training loss
had looked fine. Had we judged the model by its loss alone, we would
have reported a success and shipped a generator worse than the
baseline.

**What this won't fix on its own**: rhythm beyond 16th-note
quantisation, dynamics, articulation, lyric setting, large-scale
form. All of these need either richer tokenisation, explicit
conditioning (a key, a chorale text, a chord sequence to harmonise),
or training data with the relevant variation.

```
python3 data/make_chorales.py
transformer_chorales train    data/chorales  chorale_model  40
transformer_chorales generate chorale_model  out.mid  256  1.0  1
transformer_chorales markov   data/chorales  markov.mid  256  1
```


## 6. What this shows

The dense classifier, the convnet, the audio VAEs and the
differentiable synthesiser each did something that classical signal
processing or classical statistics already does, sometimes better.
This is the first example in the book where the classical baseline is
measured on the same data and loses clearly, and where the reason is
the architecture: a per-voice Markov chain cannot represent an
agreement between voices, and a model that attends across the whole
texture can.

The model is small, the corpus is small, and the output is a student
exercise in the style, not a chorale anyone would mistake for the
real thing. The claim is narrower and, we think, more useful: with
nothing but the layers built in this book, running on a laptop CPU
for eight minutes, a network learns a joint, structured property of
four-voice music from examples alone.

The next note takes the same model from symbols to sound. Its tokens
are no longer notes but the atoms of a matching-pursuit decomposition
[@mallat1993mp] of recorded audio, so that what the transformer
writes can be heard directly.


## Further reading

Generating Bach chorales with neural networks has a long record;
[@boulanger2012modeling] established the chorales as a benchmark for
polyphonic sequence models, and DeepBach [@hadjeres2017deepbach] and
Coconet [@huang2017coconet] are the best-known systems built for
them. The decoder-only transformer trained to predict the next token
is the design of [@radford2018gpt], and its application to music with
long-range structure is [@huang2019musictransformer]. The corpus
comes from music21 [@cuthbert2010music21].
