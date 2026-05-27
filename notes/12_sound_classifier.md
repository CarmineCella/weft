# 12 — Musical instrument classifier (SOL)

Putting the audio pipeline to work. SOL_flat is ~5000 WAV files, each
a single note from an orchestral instrument; the filename's first
dash-delimited token is the instrument class. The job is to predict
the instrument from the audio.

This is the first weft example that builds its own dataset from raw
files. Everything before this used either embedded data (IRIS) or a
purpose-built format (MNIST IDX); SOL is "a folder of audio files",
which is far closer to what real ML projects look like.

It's also the first example split across **two tools**.

---

## 1. Why two tools

Feature extraction is the slow part. Each WAV file gets framed,
windowed, FFTed several dozen times, then (for MFCC) run through the
mel filterbank and DCT. Across 5000 files this takes a couple of
minutes on a laptop. The actual neural-network training on the
extracted features takes a few seconds at most.

If we kept everything in one binary, every iteration on the classifier
(tweak an architecture, change a learning rate, try a different
dropout rate) would re-do the slow feature extraction first. Wasteful.

So the workflow splits:

```
extract_features <data_dir> <output.feat> [logmag|mfcc]   # slow, once
sound_classifier <feature_file>                            # fast, iterate
```

You run the extractor once per (dataset, feature_type) combination,
then iterate on the model as many times as you like. This is the
standard ETL-then-train pattern that essentially every real ML
project ends up with.

---

## 2. The cache format

`FeatureCache.h` provides `save_features` and `load_features`. The
file format is small and custom:

```
"WFED"  magic                          (weft feature dataset)
uint32  version = 1
uint32  feature_dim
uint32  n_samples
uint32  n_classes
length-prefixed string  feature_type   ("mfcc", "logmag", ...)
length-prefixed strings  class_names[] (n_classes of them)
float[feature_dim * n_samples]         column-major
int32[n_samples]                       labels
```

Header integers are little-endian (assembled byte-by-byte, so the
metadata is host-endian-agnostic). The bulk float and int32 payload
is native byte order — if you extract on one machine and load on a
different-endianness machine, the data will be scrambled. Don't do
that.

The feature_type label is stored in the file so the classifier doesn't
need a separate CLI flag — it picks its architecture automatically
based on what the cache says was extracted. Sane defaults beat
duplicating configuration across tools.

---

## 3. extract_features: from WAVs to a cache file

Three things this tool does:

1. **Scan and sort.** `std::filesystem::directory_iterator` (C++17)
   walks the data dir. `.wav` files go into a vector; we sort it for
   determinism so the train/test split is reproducible.

2. **Parse the class.** The filename stem split on the first dash
   gives the class name. The map `class_to_id` grows organically as
   new prefixes appear, exactly the pattern you sketched out.

3. **Extract and cache.** For each file: `load_wav` → either
   `logmag_spectrum` or `mfcc`, accumulate into a matrix and a label
   vector. Save the result.

Failed files (bad WAV header, unsupported format) are logged (first
few only, then suppressed) and skipped. We don't want one malformed
file to abort processing of thousands of good ones.

The CLI is:

```
extract_features SOL_flat sol_mfcc.feat    mfcc
extract_features SOL_flat sol_logmag.feat  logmag
```

After both, you have two cache files and you can train the classifier
on either.

---

## 4. sound_classifier: from cache file to model

This is now a pure ML loop. Load the cache, do train/test split,
standardise, build network, train.

```cpp
auto cache = load_features(argv[1]);
auto split = train_test_split(cache.X, Y, 0.2f, /*seed=*/1);
Standardizer<float> scaler;
auto X_tr = scaler.fit_transform(split.X_train);
auto X_te = scaler.transform   (split.X_test);
```

20% test, fixed seed 1. Standardiser fit on train only — fitting on
the full dataset would leak test statistics into preprocessing.

The architecture is picked from `cache.feature_type`:

- **logmag** (2049 features): `Dense(2049, 256) → ReLU → Dropout(0.3)
  → Dense(256, 128) → ReLU → Dropout(0.3) → Dense(128, K) → Softmax`
- **mfcc** (13 features): `Dense(13, 64) → ReLU → Dropout(0.2) →
  Dense(64, K) → Softmax`

There's no point putting a 256-unit hidden layer in front of 13 input
features, and no point trying to learn 2049-dim envelopes with a
64-unit hidden. Two presets sized to their feature's information
content.

Then a standard 50-epoch loop with Adam (lr=1e-3), batch 32,
`train()`/`eval()` switching so dropout fires only during training.

---

## 5. What to expect from the two features

Rough predictions for the full SOL dataset (~5000 files, ~14
instrument classes):

- **MFCC**: 80-90% test accuracy. Training takes seconds. The
  compactness limits the ceiling — 13 numbers can't fully describe an
  oboe — but it captures enough to separate most instruments well.
- **logmag**: 85-95% test accuracy. Training takes ~30 seconds.
  Higher-resolution features let the network exploit fine details
  (specific formant peaks, harmonic decay rates) that MFCC averages
  away.

The pitch-confounding caveat from note 11 applies to both: F0
dominates the spectrum, and instruments have overlapping pitch
ranges, so some confusions are likely between acoustically similar
classes at overlapping pitches.

### A teaching example about overfitting

If you run this on a tiny dataset (a few dozen files), **the logmag
setup overfits catastrophically**. The first layer's 524K parameters
need many more examples than that to learn a generalisable mapping.
MFCC, with only a few thousand parameters total, handles small
datasets much better.

This is the classic capacity-vs-data tradeoff in microcosm. Logmag
has the expressiveness to fit complex patterns; it just needs enough
training data to constrain that expressiveness toward generalisable
patterns rather than memorising specific examples. Dropout helps but
can't manufacture data out of nothing.

Real SOL is large enough that this isn't a problem. If you want to
experiment with small subsets, stick to MFCC.

---

## 6. Things we deliberately didn't do

- **No data augmentation.** Time-stretching, pitch-shifting, and
  adding noise are common in audio ML. Would help generalisation.
- **No per-class weighting.** If some instruments have many more
  samples than others, the loss is dominated by majority classes.
  Class-balanced loss or stratified sampling fixes this.
- **No confusion matrix.** Knowing the final accuracy is one number;
  knowing *which* instruments get confused with which is much more
  diagnostic. Easy to add: a K×K matrix counting predicted-vs-true
  pairs across the test set.
- **No constant-Q / chroma features.** Either would handle the
  pitch-confounding more cleanly than mel-on-linear-FFT. More
  complexity than we need for a first pass.
- **No 1D/2D ConvNet over the spectrogram.** Modern audio classifiers
  use convolutional layers over the time-frequency representation
  rather than averaging frames into one vector. The next big weft
  milestone.
- **No incremental cache updates.** Adding new WAVs means re-running
  the full extractor. For SOL this is fine; for streaming or
  online-learning scenarios you'd want append support.

The current setup is enough to demonstrate that everything we built —
WAV I/O, FFT, windowing, mel filterbank, DCT, Adam, dropout,
train/eval mode, standardisation, a small binary serialisation
format — composes into a working end-to-end pipeline you can
actually iterate on.
