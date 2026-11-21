# 14 — Loading audio and extracting features

Feeding audio into the classifier needs a way to read sample data from
a file, and a way to turn an arbitrary-length sample stream into a
fixed-length feature vector that summarises its timbre well enough for
a classifier to learn from.

This chapter covers both, in pipeline order: WAV parsing, then framing
and windowing, then the two features we expose (log-magnitude spectrum
and MFCC).

---

## 1. The WAV format and our reader

WAV is a RIFF (Resource Interchange File Format) container, a file
built out of tagged chunks. Every chunk has the same structure:

- a 4-byte FourCC identifier (e.g. `RIFF`, `fmt `, `data`; `fmt `
  includes the trailing space)
- a 4-byte little-endian size of the chunk body
- the body itself

A minimal valid WAV is a single outer `RIFF` chunk wrapping the whole
file. Its body starts with the 4-byte string `WAVE`, then contains
sub-chunks. The ones we need are:

- `fmt `: format metadata (PCM vs float vs compressed, number of
  channels, sample rate, bits per sample)
- `data`: the samples

A naive parser would assume `fmt ` is followed by `data` and read
both at fixed offsets. In practice, WAV files often contain other
metadata chunks between them: `LIST` carrying ID3-style track info,
`JUNK` chunks inserted for alignment, and more. Our reader walks every
sub-chunk in order, processes `fmt ` and `data` when it sees them, and
skips anything else by seeking past its body. This takes a few extra
lines and makes the parser robust to whatever editor produced the
file.

### Endianness

RIFF is a Microsoft format that originated on x86, and the spec is
explicit: all multi-byte values are little-endian. To keep the parser
correct regardless of the host, we assemble multi-byte fields byte by
byte rather than reading directly into a struct or integer:

```cpp
std::uint32_t u =  std::uint32_t(b[0])
                | (std::uint32_t(b[1]) <<  8)
                | (std::uint32_t(b[2]) << 16)
                | (std::uint32_t(b[3]) << 24);
```

99% of the machines in use in 2026 are little-endian and could read
directly, but the explicit reconstruction takes the same number of
lines and also works on big-endian hosts such as a SPARC, a PowerPC
G5, or a big-endian embedded processor.

### What we accept

The `fmt ` chunk's first field is a format code that identifies how to
interpret the `data` chunk:

- `1`: linear PCM (raw signed integer samples)
- `3`: IEEE float samples
- `6, 7`: A-law / μ-law (telephony compression)
- `0xFFFE`: WAVE_FORMAT_EXTENSIBLE (anything else, disambiguated by a
  GUID at the end of the fmt chunk)
- many more: various compressed codecs

We accept only code `1` with 16 bits per sample. SOL files are 16-bit
PCM, so that covers the use case. Every additional code would be a
dispatch path that needs its own decoding. Float WAVs need scaling by
the right factor and no conversion from int. 24-bit files need three
bytes per sample with sign extension. WAVE_FORMAT_EXTENSIBLE needs
reading the GUID and matching it against the PCM GUID. These would be
parsers for cases we don't need. If the file isn't what we expect, the
loader throws.

Samples are signed 16-bit integers in [−32768, +32767]. We divide by
32768 to land in [−1, 1], the conventional range for downstream DSP.

### Stereo to mono

If the file is stereo, the `data` chunk holds interleaved L, R, L, R,
... samples. We average them to produce a mono signal. More elaborate
handling (channel selection, mid-side decoding) wouldn't help
instrument classification, because the spectral envelope of a
sustained note is essentially identical in both channels.

`load_wav` returns a `WavData` struct with the mono samples as a
`std::vector<float>` and the sample rate. The feature extractors need
nothing else.

---

## 2. The framing pipeline

The FFT gives us a spectrum, but a single FFT of a whole audio file is
not useful. A 1-second piano note isn't a stationary signal. It has an
attack with broadband energy, then a steady tone with harmonics, then
a decaying release. A single FFT over the whole note mixes all of
these together.

We want one feature vector per file that summarises the character of
the sound well enough to classify. We get it by analysing short
frames, then averaging those analyses with energy weighting so that
the silent decay doesn't dilute the spectrum of the attack.

The structure is the same regardless of which feature we compute:

```
for each frame at offsets 0, hop, 2*hop, ...:
    multiply frame by Hann window
    compute frame energy = sum of squared windowed samples
    compute frame-level feature (this is what differs)
    accumulate feature, weighted by energy
divide accumulated feature by total energy
```

The parameters are `frame_size` (typically 4096 samples at 44.1 kHz,
around 93 ms), `hop_size` (typically half of frame_size, giving 50%
overlap), and the per-frame feature function.

### Why a window function?

If we chop the signal into 4096-sample chunks and FFT them, each chunk
is in effect multiplied by a rectangular window. A rectangular window
has poor frequency-domain behaviour: a pure sine at frequency `f`
shows up in the FFT as a wide sinc-shaped smear instead of a narrow
peak. This is **spectral leakage**, and it degrades everything
downstream.

The Hann window, `w[i] = 0.5 * (1 - cos(2π i / (N-1)))`, tapers
smoothly to zero at both endpoints. Its highest side-lobe is about
−31 dB, against −13 dB for a rectangular window, and the side-lobes
fall off much faster, so leakage becomes a small effect instead of a
dominant one. Other windows (Hamming, Blackman, Kaiser) have different
tradeoffs between main-lobe width and side-lobe height. Hann is the
standard default and is adequate here.

### Why energy-weighted averaging?

A note's character is in its loud parts. The release of a piano note,
where the energy has decayed by 40 dB, carries little instrument
information. It is mostly room reverb and noise. With a plain mean
across frames, those quiet tails would dilute the signal.

Weighting each frame by its energy (`sum of squared windowed samples`)
puts essentially all the contribution into the loud frames. Frames
below an energy threshold are skipped entirely, which also handles
leading and trailing silence.

---

## 3. Log-magnitude spectrum

This is the simpler of the two features. For each frame:

1. FFT the windowed frame.
2. Take magnitudes of the first `N/2 + 1` bins.
3. Take `log(1 + |X[k]|)` to compress the dynamic range.

The `1 +` inside the log avoids `log(0)` for silent bins. Since we
already skip whole silent frames, this is an extra precaution.

The output dimension is `N/2 + 1`. For N=4096 that's 2049 features per
file, a high-resolution description of the spectral envelope.

The log compression is the important step. A typical instrument's
fundamental might be 10⁴ times larger than its 10th harmonic, and a
neural network whose first layer is fed the raw magnitudes will
respond mostly to the fundamental and effectively ignore the rest.
With the log, that 10⁴ ratio becomes a difference of
`log(10⁴) ≈ 9.2`, and the high harmonics contribute to the feature
vector on a comparable scale.

This feature is large but expressive, and a good baseline.

---

## 4. MFCC: Mel-Frequency Cepstral Coefficients

MFCCs are a more compact and historically standard feature. The steps
per frame are:

1. **FFT and power spectrum.** Square the magnitudes to get power.
2. **Mel filterbank.** Apply a set of triangular filters spaced on the
   mel scale, summing the power within each filter. This produces a
   low-dimensional mel spectrum (e.g. 40 numbers per frame).
3. **Log.** Take `log` of each mel-spectrum value.
4. **DCT-II.** Apply a discrete cosine transform and keep only the
   first 13 coefficients.

We then accumulate (energy-weighted average) across frames to get one
13-dimensional vector per file.

### The mel scale

Humans don't perceive pitch linearly. The difference between 100 Hz
and 200 Hz sounds like a full octave, while the difference between
5000 Hz and 5100 Hz is barely perceptible. The mel scale is a
perceptual remapping in which equal mel distances approximate equal
perceived pitch distances:

```
mel(hz)  =  2595 * log10(1 + hz / 700)
```

So 700 Hz → 781 mel, 1400 Hz → 1238 mel (not 1562; the spacing
shrinks as frequency grows).

For audio classification, putting features on the mel scale means
that we care more about distinguishing 200 Hz from 400 Hz than
8000 Hz from 8200 Hz, which matches what is informative about
instrument timbre.

### The mel filterbank

Pick a number of filters (40 is standard), space their centre
frequencies evenly in mel space across the frequency range, and make
each filter a triangle that is 1 at its centre and 0 at the centres of
its neighbours. Sum the FFT power within each filter to get that
filter's energy.

The output is a 40-dimensional summary of the power spectrum, biased
toward lower frequencies (where the filters are narrower and more
numerous). Human hearing has the same bias.

### Why the DCT at the end?

The 40 log-mel-energies are strongly correlated: adjacent filters
overlap, and slow trends in the spectrum show up across many bins.
The DCT changes basis to one where slow trends are captured by the
first few coefficients and faster oscillations by later ones. Keeping
the first 13 discards the high-frequency detail (which is mostly
noise) and keeps the smooth shape (which is mostly timbre).

Modern audio neural networks often skip the DCT and feed the
log-mel-spectrogram straight to a network, which learns the
decorrelation itself. We do the DCT here because (a) it makes the
feature 13-dim instead of 40-dim, which keeps the downstream MLP
small, and (b) it is of teaching interest: the cepstrum (DCT of log
spectrum) is the historical foundation of audio analysis, going back
to speech recognition in the 1960s.

---

## 5. Averaged vs per-frame output

Both feature functions come in two forms. The averaged forms
(`logmag_spectrum`, `mfcc`) collapse the whole signal into a single
energy-weighted vector. The per-frame forms (`logmag_spectrum_frames`,
`mfcc_frames`) keep one vector per non-silent frame and return a list
of them.

The windowing, FFT, mel filterbank and DCT are the same. The only
difference is whether the per-frame results are averaged at the end or
kept separate.

When to use which:

- **Averaged**: when you want one feature per file, the data is
  short enough that an average over the whole sound is meaningful,
  and the file count itself is enough training signal. It is compact
  and fast.
- **Per-frame**: when you want to (a) treat each frame as its own
  training example so the network sees temporal variation, (b) reuse
  the output for a per-frame application like an autoencoder that
  reconstructs frame-by-frame, or (c) aggregate downstream predictions
  per file by averaging softmax outputs across frames at eval time
  (ensembling).

For TinySOL, per-frame extraction gives about 135 times as many
training rows (the notes average about six seconds, which at 44.1 kHz
with frame 4096 and hop 2048 is ~135 frames per file). The downstream
classifier needs to (a) split by file_id rather than by row, so that
frames from the same file do not leak across train and test, and
(b) aggregate per file at eval time. Both are handled by
`group_train_test_split` in `Data.h` and per-file aggregation in the
classifier.

Per-frame is also what the audio VAE of [chapter 19](19_audio_vae.md)
trains on: each frame is one training example, the encoder learns a
compact representation of plausible instrument spectra, and the
decoder inverts it.

The per-frame functions skip silent frames (energy below a small
threshold) so we don't train on noise from leading/trailing silence.
The averaged version effectively does the same thing through energy
weighting.

---

## 6. Which feature to choose?

For the instrument classifier, both are worth trying:

- **logmag_spectrum** keeps everything: 2049 features per file. With
  1,532 TinySOL files and a Dense(2049, 256) first layer, that's around
  500K weights from the input alone. There is a risk of overfitting;
  dropout and standardisation help.

- **mfcc** is a much smaller representation: 13 features per file.
  The classifier is small, and Dense(13, 64) → Dense(64, K) is enough.
  It trains much faster. It likely loses some discrimination ability
  because of the compression.

Both are exposed in the example: the feature type is the third argument
of `extract_features` (`logmag` or `mfcc`).
[Chapter 15](15_sound_classifier.md) measures both. With one vector
per file, logmag is about 2 percentage points more accurate than MFCC,
and MFCC trains in a fraction of a second. Per-frame mode lifts both
to about 98% through file-level aggregation at evaluation time.

---

## 7. What is left out

A full audio pipeline usually adds several things that we skip:

- **Pre-emphasis filter** (a simple high-pass that flattens the
  spectrum before framing). Adds ~1% accuracy on speech tasks; less
  clearly useful for music.
- **Delta and delta-delta MFCCs** (time derivatives of MFCC across
  frames). Captures dynamics; not relevant when we average everything
  to a single vector anyway.
- **Constant-Q transform** (logarithmically-spaced frequency bins
  instead of mel triangles on linear bins). Stronger pitch invariance
  than mel; more complex to implement.
- **Spectrogram features for ConvNets** (no averaging; the time axis
  is kept, giving a 2D image). Used in [chapter 23](23_convvae.md),
  a convolutional VAE on mel-spectrogram patches.
- **Float, 24-bit, or compressed WAVs.** Audio libraries add the
  dispatch logic. TinySOL does not need it.

For TinySOL classification, 16-bit PCM input and an energy-weighted
log-mag spectrum or MFCC output are enough to get a representative
result.


## Further reading

The mel scale of pitch is [@stevens1937mel], and mel-frequency cepstral
coefficients were introduced for speech recognition by [@davis1980mfcc].
The classic reference on window functions is [@harris1978windows]. For
the short-time Fourier transform as used in audio, see
[@smith2011spectral].
