# 12 — Musical instrument classifier (SOL)

Putting the audio pipeline to work. SOL_flat is ~5000 WAV files, each
a single note from an orchestral instrument; the filename's first
dash-delimited token is the instrument class. The job is to predict
the instrument from the audio.

This is the first weft example that builds its own dataset from raw
files, and it's split across **two tools** so the slow feature-
extraction step doesn't repeat on every classifier iteration.

---

## 1. Why two tools

Feature extraction is the slow part. Each WAV file gets framed,
windowed, FFTed several dozen times, then (for MFCC) run through the
mel filterbank and DCT. Across 5000 files this takes a couple of
minutes on a laptop. The actual neural-network training on the
extracted features takes a few seconds at most.

If we kept everything in one binary, every iteration on the classifier
would re-do the slow extraction first. Wasteful.

So the workflow splits:

```
extract_features <data_dir> <output.feat> [logmag|mfcc] [averaged|per_frame]
sound_classifier <feature_file>
```

You run the extractor once per (dataset × feature_type × mode)
combination, then iterate on the model as many times as you like. This
is the standard ETL-then-train pattern that essentially every real ML
project ends up with.

---

## 2. Two extraction modes

### Averaged (one feature per file)

The original mode: STFT, energy-weighted average across frames, one
fixed-length vector per file. Compact, fast, and the baseline result.
Network architecture is sized to the feature dimension.

### Per-frame (one feature per non-silent frame)

For each WAV file we keep every non-silent STFT frame as its own
training example. Same feature dimension as averaged mode (2049 for
logmag, 13 for MFCC), but ~20× more rows.

Why it matters:

- More training data per file → the network's parameters are better
  constrained, less overfitting.
- The network can learn from the instrument's temporal evolution
  (attack vs sustain vs release) rather than only seeing the time-
  averaged spectrum.
- At inference, predictions are aggregated across all frames of a
  test file by averaging softmax outputs and taking argmax. This is
  ensembling and typically buys 5-10 percentage points on its own.
- The same per-frame infrastructure is what we'll need for the
  autoencoder / morphing work.

Trade-off: cache file is ~20× larger, training is ~20× more
iterations per epoch (though each iteration is the same cost since
the architecture is unchanged).

For real SOL the averaged mode plateaus around 75% test accuracy with
log-mag features; per-frame typically pushes that into the 85%+
range, with much of the gap coming from per-file ensembling rather
than the larger training set.

---

## 3. The file-leakage problem (and the fix)

Per-frame mode introduces a subtle issue: frames from the same file
are not independent. They share most of their spectral character.
If you split frames at random into train/test, frames from the same
SOL file end up on both sides, and the network can essentially
"memorise" specific files at frame granularity. Test accuracy looks
inflated.

The fix is to split **by file_id**, not by frame: choose 20% of files
to be test, all of their frames go to test, none of their frames go
to train. The cache stores a `file_id` per row exactly for this
purpose, and `group_train_test_split` in `Data.h` does the file-aware
split.

For averaged mode this distinction doesn't matter — each row is its
own file by construction — but the same function handles both cases
uniformly (in averaged mode file_ids are `{0, 1, 2, ...}` and the
behaviour collapses to a regular random split).

---

## 4. Per-file aggregation at eval time

Once the network is trained, evaluating per-frame mode means:

1. Forward-pass all test frames through the network in batches.
2. Group the resulting softmax outputs by `file_id`.
3. Average the softmax outputs across frames of each file.
4. Argmax on the averaged vector to get a per-file prediction.
5. Compare to the file's true label.

This is a textbook example of test-time ensembling. Each frame gives
a (slightly noisy) probability distribution over classes; averaging
across all frames of a file cancels out frame-level noise and gives a
much sharper file-level prediction.

The reported "test acc (per file)" reflects this aggregation.

---

## 5. Confusion matrix

After training, we print a `K × K` confusion matrix over the test
set: rows are true class, columns are predicted class. The diagonal
shows correct counts; off-diagonal entries show specific confusions.

This is much more informative than a single accuracy number. With
~15 instruments you see, for example:

- Violin and viola confused with each other at high rate → structural
  problem with this representation (their overlapping pitch ranges +
  similar spectral envelopes are nearly indistinguishable in
  averaged-over-time features)
- Brass instruments cross-confused but not confused with strings →
  the network learned the broad timbre family but is fuzzy on
  individuals
- Scattered random off-diagonals → mostly model noise

For SOL specifically, the dominant confusions are within instrument
families. That's both diagnostic and pedagogically interesting: a
human listener would have the same trouble with isolated notes.

---

## 6. The cache format (v2)

`FeatureCache.h` provides `save_features` and `load_features`. The
file format:

```
"WFED"  magic                          (weft feature dataset)
uint32  version = 2
uint32  feature_dim
uint32  n_samples (or n_frames in per_frame mode)
uint32  n_classes
uint32  per_frame (0 or 1)
length-prefixed string  feature_type   ("mfcc", "logmag", ...)
length-prefixed strings  class_names[] (n_classes of them)
float[feature_dim * n_samples]         column-major
int32[n_samples]                       labels
int32[n_samples]                       file_ids
```

Header integers are little-endian (assembled byte-by-byte, host-
endian-agnostic). The bulk payload is native byte order — if you
extract on one machine and load on a different-endianness machine,
the floats will be scrambled. Don't do that.

Version was bumped from 1 to 2 to add `per_frame` and `file_ids`.
v1 caches can't be loaded; just re-run `extract_features`.

---

## 7. What to expect from the four combinations

| feature | mode      | typical test accuracy | training time |
|---------|-----------|-----------------------|---------------|
| mfcc    | averaged  | 70-80%                | seconds       |
| mfcc    | per_frame | 80-88%                | < 30s         |
| logmag  | averaged  | 75-85%                | ~30s          |
| logmag  | per_frame | 85-92%                | ~3 min        |

(Per-frame logmag is the slowest because of the 2049-dim Dense first
layer × 20× more frames. Still very feasible on a laptop.)

The gap from averaged → per_frame is mostly per-file ensembling
working as intended. The gap from mfcc → logmag is the higher-
resolution feature giving the network more to work with.

---

## 8. What's still not in here

- **CQT (Constant-Q Transform)** — logarithmically-spaced bins for
  better pitch invariance. Genuinely worth adding; would slot in as a
  third feature type alongside `logmag` and `mfcc`.
- **Per-class weighting** — if some instruments have many more samples
  than others, the loss is dominated by majority classes.
- **Data augmentation** — pitch-shifting, time-stretching, additive
  noise. Standard moves to lift accuracy further.
- **Spectrogram-based ConvNet** — instead of averaging-or-aggregating
  frame-level features, feed the full (frames × bins) spectrogram to
  a 2D ConvNet. The biggest jump available, and the natural way to
  handle time in audio. On the roadmap once ConvNets are built.

The current pipeline composes everything else we have — WAV I/O, FFT,
windowing, mel filterbank, DCT, Adam, dropout, train/eval mode,
standardisation, group-aware splitting, a small binary serialisation
format — into a working end-to-end classifier you can iterate on.
