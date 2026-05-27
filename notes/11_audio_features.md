# 11 — Loading audio and extracting features

Two things are needed to feed audio into the classifier: a way to read
sample data from a file, and a way to turn an arbitrary-length sample
stream into a fixed-length feature vector that summarises its
timbre well enough for a classifier to learn from.

This note covers both, in pipeline order: WAV parsing, then framing
and windowing, then the two feature flavours we expose (log-magnitude
spectrum and MFCC).

---

## 1. The WAV format and our reader

WAV is a RIFF (Resource Interchange File Format) container — a file
built out of tagged chunks. Every chunk has the same three-part
structure:

- a 4-byte FourCC identifier (e.g. `RIFF`, `fmt `, `data` — yes, with
  the trailing space on `fmt `)
- a 4-byte little-endian size of the chunk body
- the body itself

A minimal valid WAV is a single outer `RIFF` chunk wrapping the whole
file. Its body starts with the 4-byte string `WAVE`, then contains
sub-chunks. The two we care about:

- `fmt ` — format metadata: PCM vs float vs compressed, number of
  channels, sample rate, bits per sample
- `data` — the actual samples

The naive parser would assume `fmt ` is followed by `data` and read
both at fixed offsets. In practice, real-world WAVs often interleave
metadata chunks: `LIST` carrying ID3-style track info, `JUNK` chunks
inserted for alignment, and more. Our reader walks every sub-chunk in
order, processes `fmt ` and `data` when it sees them, and skips
anything else by seeking past its body. This is a few extra lines and
makes the parser robust to whatever editor produced the file.

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

99% of the machines anyone runs in 2026 are little-endian and could
read directly, but the explicit reconstruction is the same number of
lines and works everywhere — including, hypothetically, on a
big-endian SPARC, on a PowerPC G5 nobody has touched in fifteen years,
or on whatever embedded thing decides to be old-school in the future.

### What we actually accept

The `fmt ` chunk's first field is a format code that identifies how to
interpret the `data` chunk:

- `1`      — linear PCM (raw signed integer samples)
- `3`      — IEEE float samples
- `6, 7`   — A-law / μ-law (telephony compression)
- `0xFFFE` — WAVE_FORMAT_EXTENSIBLE (anything else, disambiguated by a
             GUID at the end of the fmt chunk)
- many more — various compressed codecs

We accept only code `1` with 16 bits per sample. SOL files are 16-bit
PCM, so that covers the use case. Every additional code we'd accept is
a dispatch path that needs its own decoding — for float WAVs, scale by
the right factor and no conversion from int; for 24-bit, three bytes
per sample with sign extension; for WAVE_FORMAT_EXTENSIBLE, read and
match against the PCM GUID. We'd be writing parsers for cases we don't
need. If the file isn't what we expect, the loader throws.

Samples are signed 16-bit integers in [−32768, +32767]. We divide by
32768 to land in [−1, 1] — the conventional range for downstream DSP.

### Stereo to mono

If the file is stereo, the `data` chunk holds interleaved L, R, L, R,
... samples. We average them to produce a mono signal. Anything more
sophisticated (channel selection, mid-side decoding) wouldn't help
instrument classification: the spectral envelope of a sustained note
is essentially identical in both channels.

The reader returns a `WavData` struct with the mono samples as a
`std::vector<float>` and the sample rate. That's everything the
feature extractors need.

---

## 2. The framing pipeline

The FFT gives us a spectrum, but a single FFT of a whole audio file is
useless. A 1-second piano note isn't a stationary signal — it has an
attack with broadband energy, then a steady tone with harmonics, then
a decaying release. Averaging the whole thing into one FFT smears all
of that together.

What we want is *one feature vector per file* that summarises the
character of the sound well enough to classify. We get there by
analysing short frames, then averaging those analyses with energy
weighting so the silent decay doesn't pollute the spectrum of the
attack.

The structure is the same regardless of which feature we compute:

```
for each frame at offsets 0, hop, 2*hop, ...:
    multiply frame by Hann window
    compute frame energy = sum of squared windowed samples
    compute frame-level feature (this is what differs)
    accumulate feature, weighted by energy
divide accumulated feature by total energy
```

Three parameters: `frame_size` (typically 4096 samples at 44.1 kHz,
around 93 ms), `hop_size` (typically half of frame_size, giving 50%
overlap), and the per-frame feature function.

### Why a window function?

If we just chop the signal into 4096-sample chunks and FFT them, each
chunk effectively gets multiplied by a rectangular window. A
rectangular window has terrible frequency-domain behaviour: a pure
sine at frequency `f` shows up in the FFT as a wide sinc-shaped smear,
not a clean peak. This is **spectral leakage**, and it ruins
everything downstream.

The Hann window — `w[i] = 0.5 * (1 - cos(2π i / (N-1)))` — tapers
smoothly to zero at both endpoints. The frequency-domain side-lobes
are roughly 30 dB lower than a rectangular window's, which means
leakage becomes a small effect instead of a dominant one. There are
other windows (Hamming, Blackman, Kaiser) with different tradeoffs
between main-lobe width and side-lobe height; Hann is the standard
default and fine for our purposes.

### Why energy-weighted averaging?

A note's character is in its loud parts. The release of a piano note,
where the energy has decayed by 40 dB, isn't carrying much instrument
information — it's mostly room reverb and noise. If we did a plain
mean across frames, those quiet tails would dilute the signal.

Weighting each frame by its energy (`sum of squared windowed samples`)
puts essentially all the contribution into the loud frames. Frames
below an energy threshold are skipped entirely, which also handles
leading and trailing silence.

---

## 3. Log-magnitude spectrum

The simpler of the two features. For each frame:

1. FFT the windowed frame.
2. Take magnitudes of the first `N/2 + 1` bins.
3. Take `log(1 + |X[k]|)` to compress the dynamic range.

The `1 +` inside the log lets us handle silent bins without `log(0)`;
since we already skip whole silent frames, this is belt-and-braces.

Output dimension: `N/2 + 1`. For N=4096 that's 2049 features per
file — a high-resolution snapshot of the spectral envelope.

The log compression is the load-bearing trick. A typical instrument's
fundamental might be 10⁴ times larger than its 10th harmonic, and a
neural network whose first layer is fed the raw magnitudes will pay
attention to the fundamental and effectively ignore the rest. With
log, that 10⁴ ratio collapses to a difference of `log(10⁴) ≈ 9.2`, and
the high harmonics become first-class citizens in the feature vector.

This feature is large but expressive. Good as a baseline.

---

## 4. MFCC: Mel-Frequency Cepstral Coefficients

A more compact and historically standard feature. Four steps per
frame:

1. **FFT and power spectrum.** Square the magnitudes to get power.
2. **Mel filterbank.** Apply a set of triangular filters spaced on the
   mel scale, summing the power within each filter. Produces a
   short-dimensional "mel spectrum" (e.g. 40 numbers per frame).
3. **Log.** Take `log` of each mel-spectrum value.
4. **DCT-II.** Apply a discrete cosine transform and keep only the
   first 13 coefficients.

We then accumulate (energy-weighted average) across frames to get one
13-dimensional vector per file.

### The mel scale

Humans don't perceive pitch linearly. The difference between 100 Hz
and 200 Hz sounds like a full octave; the difference between 5000 Hz
and 5100 Hz is barely perceptible. The mel scale is a perceptual
remapping that approximates "equal mel distance feels like equal
pitch distance":

```
mel(hz)  =  2595 * log10(1 + hz / 700)
```

So 700 Hz → 781 mel, 1400 Hz → 1336 mel (not 1562 — the spacing
shrinks as frequency grows).

For audio classification, putting features on the mel scale is a way
of saying "we care more about distinguishing 200 Hz from 400 Hz than
8000 Hz from 8200 Hz", which matches what's actually informative
about instrument timbre.

### The mel filterbank

Concretely: pick a number of filters (40 is standard), space their
centre frequencies evenly in mel space across the frequency range,
and make each filter a triangle that's 1 at its centre and 0 at the
centres of its neighbours. Sum the FFT power within each filter to
get that filter's energy.

The output is a 40-dimensional summary of the power spectrum, biased
toward lower frequencies (where the filters are narrower and more
numerous) — exactly the bias human ears have.

### Why the DCT at the end?

The 40 log-mel-energies are strongly correlated — adjacent filters
overlap, and slow trends in the spectrum show up across many bins.
The DCT changes basis to one where slow trends are captured by the
first few coefficients and faster oscillations by later ones. Keeping
the first 13 throws away the high-frequency detail (which is mostly
noise) and keeps the smooth shape (which is mostly timbre).

It's worth noting that modern audio neural networks often skip the
DCT entirely — they feed the log-mel-spectrogram straight to a NN and
let the network learn the decorrelation. We do the DCT here because
(a) it makes the feature 13-dim instead of 40-dim, which keeps the
downstream MLP small, and (b) it's pedagogically interesting: the
"cepstrum" (DCT of log spectrum) is the historical foundation of
audio analysis going back to speech recognition in the 1960s.

---

## 5. Which feature to choose?

For the instrument classifier, both are worth trying:

- **logmag_spectrum** keeps everything. 2049 features per file. With
  ~5000 SOL files and a Dense(2049, 256) first layer, that's around
  500K weights from the input alone. Risk of overfitting; dropout
  and standardisation help.

- **mfcc** is a much smaller representation. 13 features per file.
  The classifier is tiny — Dense(13, 64) → Dense(64, K) is enough.
  Much faster to train. Likely loses some discrimination ability
  because of the compression.

We'll expose both in the example and let you toggle between them with
a CLI flag. My guess: logmag will be 2-5 percentage points more
accurate, but MFCC will train in seconds. Both should comfortably
clear chance accuracy by a wide margin.

---

## 6. What's not in here (yet)

A real audio pipeline usually adds several things we're skipping:

- **Pre-emphasis filter** (a simple high-pass that flattens the
  spectrum before framing). Adds ~1% accuracy on speech tasks; less
  clearly useful for music.
- **Delta and delta-delta MFCCs** (time derivatives of MFCC across
  frames). Captures dynamics; not relevant when we average everything
  to a single vector anyway.
- **Constant-Q transform** (logarithmically-spaced frequency bins
  instead of mel triangles on linear bins). Stronger pitch invariance
  than mel; more complex to implement.
- **Spectrogram features for ConvNets** (don't average at all — keep
  the time axis as a 2D image). The next big architectural milestone
  for weft.
- **Float, 24-bit, or compressed WAVs.** Real-world audio libraries
  add the dispatch logic; for SOL specifically we can get away
  without it.

For SOL classification, 16-bit PCM in plus energy-weighted log-mag
spectrum or MFCC out is enough to get a representative result.
