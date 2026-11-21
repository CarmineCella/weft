# 29 — A transformer that writes sound

[Chapter 27](27_transformer_chorales.md) trained a transformer on the
notes of chorales. [Chapter 28](28_matching_pursuit.md) showed how to
write a recording as a list of atoms and how to turn a list of atoms
back into sound. This chapter combines them. The transformer is trained
on the atom lists of a few recordings. Sampling from it produces a new
atom list, and rendering that list produces audio.

The network does not output samples or spectra. It chooses the next
atom: when it starts, how long it is, its frequency, its level and its
phase. The sound is the sum of the windowed sinusoids it chose.

The model is the same class as in chapter 27, `TokenTransformer<T>`.
Only the tokens are different. Most of this chapter is about the
tokens, because that is where the decisions are.


## 1. From atoms to tokens

A decomposition gives the atoms in order of strength. A sequence model
needs them in order of time. We keep the strongest 2,000 atoms of each
sound, which chapter 28 found to be enough to keep a sound
recognisable, and sort them by the time of their centre. Each atom
then becomes five tokens:

| Field | Meaning | Values |
|---|---|---|
| `DT` | time since the centre of the previous atom | 128 |
| `SCALE` | which of the four window lengths | 4 |
| `FREQ` | frequency, 60 steps per octave above 20 Hz | 608 |
| `AMP` | level, in steps of 1.5 dB below the loudest atom | 48 |
| `PHASE` | phase of the cosine at the centre of the atom | 16 |

With a start token and an end token the vocabulary has 806 tokens. A
sound of 2,000 atoms is a sequence of 10,002 tokens.

**Time.** Most atoms follow the previous one within a few
milliseconds. The first 96 values of `DT` are exact multiples of 32
samples (0.7 ms), up to 70 ms. A sound also contains pauses, so the
remaining 32 values grow geometrically up to 3 seconds. Section 2
explains why the second range is needed.

**Frequency.** Each window length has its own frequency bins, from 64
bins for the shortest window to 4,096 for the longest. A separate
token for every bin of every length would give a vocabulary of
thousands of tokens, most of them seen only a few times. One
logarithmic grid shared by all lengths keeps the vocabulary small. On
decoding, a `FREQ` token is mapped to the nearest bin of the atom's
window length.

**Phase.** An atom stores its phase at its first sample. The token
stores the phase at the centre of the atom:
`phase_centre = phase_start + pi * bin`. The frequency grid can move
an atom to a neighbouring bin, and the window is largest at the
centre, so that is where the cosine must stay in place.


## 2. What the tokens cost

Before training anything we measure what the tokens lose.
`mp_generate roundtrip` converts the 2,000 atoms of a sound to tokens
and back, and renders the result. We compare its magnitude
spectrogram with that of the same 2,000 atoms rendered directly. The
figure is the energy of the difference divided by the energy of the
reference. In every column the time, frequency and level go through
the tokens. The columns differ in the phase: kept exactly, quantised
to 16 values, restored by a rule, or drawn at random.

| Sound | Exact phase | 16 phase values | Phase by a rule | Random phase |
|---|---|---|---|---|
| `cage.wav` | 8% | 9% | 47% | 46% |
| `radiohead.wav` | 6% | 8% | 41% | 41% |
| `Frank_Gamb_ori.wav` | 13% | 14% | 42% | 43% |

The grids for time, frequency and level change the spectrogram by 6
to 13%. Quantising the phase to 16 values adds about one point.

The last two columns are the reason phase is a token. Our first design
had four tokens per atom and restored the phase at synthesis. One rule
gave every atom the phase of a single cosine at its frequency,
counted from the start of the sound, so that two overlapping atoms at
the same frequency add up. The other rule drew the phase at random.
Both change the spectrogram by more than 40%, three to seven times
the cost of all the other fields together. The rule based on a single
cosine fails because an atom's frequency is known only to the nearest
bin, and the true phase of a partial advances at its true frequency.

The same measurement found a bug. In the first version `DT` had only
the linear range and gaps longer than 92 ms were cut to 92 ms. Every
pause became shorter and everything after it arrived early. Two of
the three sounds came back with a spectrogram difference of about
100%. The geometric range fixed it.


## 3. The model and its training

The model has three causal blocks with `d_model = 128`, four heads
and `d_ff = 512`, 802,214 parameters. A training window is 320
tokens, which is 64 atoms, about a third of a second of sound.

The window rules of chapter 27 apply unchanged. A window starts at a
multiple of 5 tokens, so that the position modulo 5 tells the model
which field it is writing. One window in twenty starts at the
beginning of a sound.

The corpus is three recordings of 10 to 13 seconds, 2,000 atoms each.
The last tenth of each sound, in time, is held out. That leaves 9,001
training tokens and 1,001 held-out tokens per sound.

```
mp_analyze  decompose  data/cage.wav  cage  8000
mp_generate train  model  40  2000  cage.atoms radiohead.atoms Frank_Gamb_ori.atoms
```

The corpus is very small: 27,000 tokens for 800,000 parameters. The
training log shows the consequence.

| Epoch | Training loss | Held-out loss |
|---|---|---|
| 1 | 4.15 | 3.08 |
| 5 | 2.55 | 2.54 |
| 10 | 2.41 | 2.47 |
| 20 | 2.15 | 2.40 |
| 30 | 1.98 | 2.62 |
| 40 | 1.82 | 2.73 |

After epoch 20 the training loss keeps falling and the held-out loss
rises. The model has started to memorise its three sounds. The
program saves the weights with the lowest held-out loss, here those
of epoch 20, and those are the weights used below. An epoch takes
about 10 seconds.

Two baselines are fitted on the same training tokens and scored on
the same held-out tokens, in nats per token. The unigram model uses
how often each value of a field occurs. The Markov chain predicts
each field from the same field of the previous atom. Both are told
which field each position holds, as the transformer is.

| Model | Held-out loss |
|---|---|
| Unigram | 2.71 |
| Markov chain | 2.83 |
| Transformer, epoch 20 | 2.40 |

The transformer predicts the held-out atoms better than both. The
margin is much smaller than for the chorales, where the loss fell
from 1.27 to 0.49. The Markov chain is worse than the unigram model
here because most of its transition counts are zero: with 608
frequency values and fewer than 6,000 atoms, most pairs of values
never occur.


## 4. Sampling

Generation is the loop of chapter 27 with one addition. At every step
only the tokens of the field that comes next are allowed. The
probabilities of all other tokens are set to zero before sampling.
The stream is then always a well-formed list of atoms, whatever the
model predicts.

The context is the last 320 tokens, starting at a multiple of 5.
Sampling stops when the sum of the `DT` values reaches the requested
duration.

```
mp_generate generate  model  new.wav  12  1.0  1
mp_generate generate  model  cont.wav  12  1.0  1  cage.atoms  1.0
mp_generate markov    model  markov.wav  12  1
```

The second command starts from the atoms of the first second of a
recording and lets the model continue. The third samples from the
Markov baseline. Twelve seconds of sound are about 3,000 atoms, or
15,000 forward passes, and take two to three minutes.


## 5. What the model generates

The first check is whether the model copies its training sounds. With
27,000 training tokens that is the obvious risk. The program counts
the runs of 8 consecutive generated atoms (40 tokens) that occur,
token for token, in a training sound. In every sound generated for
this chapter the count is zero. The output is not a collage of pieces
of the recordings.

The second check is how the generated atoms are organised in time.
`mp_generate stats` computes four numbers for an atom list:

- the number of atoms per second;
- the share of atoms that use one of the two longest windows;
- **continuity**: of the atoms with a long window, the share that
  follow another long atom at the same frequency within their own
  length. A sustained partial is a chain of such atoms;
- **clustering**: of the atoms with a short window, the share that
  have another short atom within 6 ms. An attack is a group of them.

| Atom list | Atoms per second | Long windows | Continuity | Clustering |
|---|---|---|---|---|
| `cage.wav` | 189 | 70% | 31% | 80% |
| `radiohead.wav` | 157 | 71% | 42% | 88% |
| `Frank_Gamb_ori.wav` | 214 | 75% | 43% | 74% |
| Transformer, temperature 1.0 (three sounds) | 250–277 | 79–81% | 39–47% | 81–83% |
| Transformer, temperature 0.8 (two sounds) | 252–349 | 92–93% | 64–65% | 79–83% |
| Markov chain (three sounds) | 153–204 | 69–70% | 26–27% | 82–86% |

The transformer at temperature 1.0 is close to the recordings in
continuity and clustering. It writes more atoms per second than any
of the three recordings, and more of them are long. At temperature
0.8 both tendencies are stronger: more than nine atoms in ten use a
long window. A lower temperature favours the most probable choice,
and the most probable atom is a long one.

The Markov chain reproduces the number of atoms per second and the
share of each window length, because those are properties of single
fields and that is all it models. Its continuity is 26%, below every
recording: it draws each frequency from the previous atom's
frequency alone, and the previous atom in time is usually part of a
different partial. The transformer sees 64 atoms and can continue a
partial that was interrupted by others. This is the measurable
difference between the two, and it is the same kind of difference as
voice crossing in the chorales.

These numbers describe structure, not sound. Whether the result is
interesting to listen to cannot be read from a table. The commands
above produce the examples.


## 6. Limitations

**The context is a third of a second.** The model sees 64 atoms. It
can continue a partial and place the atoms of an attack together. It
cannot know what happened one second earlier, so it cannot produce a
phrase or a rhythm that lasts longer than its window. A longer window
costs more: attention grows with the square of the window length.

**The corpus is three sounds.** The held-out loss turns upward after
20 epochs. A model of this size needs far more material before more
training helps. With more recordings the same program applies
unchanged.

**The output is denser than the recordings.** The model writes 250
to 280 atoms per second where the recordings have 160 to 210, and it
uses long windows more often than they do.

**Quality is bounded by the token round trip.** The rendering of
generated tokens cannot be better than the rendering of the tokens of
a real sound (section 2), which is itself a 2,000-atom approximation
of the recording (chapter 28).

**Phase is predicted like any other token.** Two overlapping atoms of
one partial add up only if their phases agree. The model predicts
each phase from its context and nothing forces the agreement.

The system has the two parts the book has built separately. Matching
pursuit and the atom renderer are classical signal processing with no
trained parameters. The transformer is a trained model with no
knowledge of sound. The tokens are the interface between them.


## Further reading

Matching pursuit is [@mallat1993mp]. The decoder-only transformer
trained to predict the next token is [@radford2018gpt]. Generating
audio sample by sample with a neural network, which this chapter
avoids, is [@vandenoord2016wavenet] and [@mehri2017samplernn].
