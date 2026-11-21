# Book experiments with weft v0.2.0

This is a fixed reproduction protocol for the refreshed MNIST, dense audio
VAE, and differentiable-synthesis examples. It uses the C++ sources from
tag `v0.2.0`. It does not rerun every experiment in the book.

## Run from the repository root

The existing download scripts provide MNIST and the Orchidea TinySOL_2020
package. The latter package contains 1,532 WAV files from 12 instruments;
it is not the 2,913-file Zenodo distribution and this is not its official
five-fold benchmark. The complete path/hash inventory is in
[v020/corpus.json](v020/corpus.json).

```sh
python3 book/experiments/run.py prepare
python3 book/experiments/run.py mnist --backend blas
python3 book/experiments/run.py vae --backend blas
python3 book/experiments/run.py synthesis --backend blas
python3 book/experiments/analyze.py
python3 book/experiments/check.py
python3 book/experiments/bundle.py
```

The `all` command performs the first four steps sequentially. Use
`--backend portable` for the standard-library implementation, or `blas`
for Accelerate on macOS / OpenBLAS on Linux. Select a C++17 compiler with
`--compiler`; the recorded runs use Apple Clang 16. Analysis/plots need
NumPy, SciPy, and Matplotlib, as does the existing book figure script.
The training programs do not need these Python packages.

The runner builds dedicated binaries and writes caches, weights, and raw
outputs beneath the ignored `outputs/book_v020/` directory. It retains
commands, stdout/stderr, durations, toolchain information, and SHA-256
hashes in `book/experiments/v020/`. It sets `VECLIB_MAXIMUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `OMP_NUM_THREADS` to 1; synthesis additionally
uses the example's workers for a batch of 14 notes. Wall times may include
contention with the other recorded runs and are not speed benchmarks.

## Decisions made before running

**MNIST.** Run seeds 1, 2, and 3, each for ten epochs with the example's
784-256-128-10 MLP, dropout 0.3, Adam 0.001, and batch 128. Each seed
controls the 54,000/6,000 development split as well as initialization,
dropout, and shuffling. Select the lowest validation-loss checkpoint
within each run, then evaluate the official 10,000-image test set once.
Report all three accuracies and their sample standard deviation. Do not
select a seed using test accuracy.

**Audio split.** Sort the TinySOL paths. Apply the synthesis example's
length filter (at least 130 control frames at hop 512). Of the accepted
notes, indices `i % 20 == 10` form validation. This gives 1,433 training
notes, 75 validation notes, and 24 excluded short notes. The VAE trains
on the same 1,433 complete training files. Frames of a validation note
are never placed in its training cache. The file lists and hashes are
part of the record. This split tests other notes of known instruments,
not unseen instruments or recording conditions.

**Dense VAE.** Run 20 epochs, no frame subsampling, 256-column batches,
32 latent dimensions, beta 1, Adam 0.001, initialization seed 1, and
reparameterization seeds starting at 1234. Use log(1 + magnitude), a
4,096-sample symmetric Hann window, hop 2,048, and 44.1 kHz audio. Use
the final checkpoint; no epoch selection is made from the three listening
notes. The reported reconstruction term is half the sum of squared
log-magnitude errors per frame, averaged over the batch, not per-bin MSE.
Applications decode the latent mean and borrow the input phase.

**Differentiable synthesis.** Run the existing 26-256-256-256-166 network
for 40 epochs with batch 14, 128-frame crops, seed 1 for model streams,
embedding seed 7, and crop/shuffle RNG 1234. Validation uses fixed crops
starting at control frame 4. Use the final epoch, fixed in advance; the
validation set is not an untouched final test. The example computes
typical generator loudness using all accepted notes; disclose that when
interpreting generated gestures. Resynthesis explicitly receives pitch
and loudness measured from the target recording. Fit the cello controls
directly for 400 iterations as a separate per-note reference, not a
globally optimal bound or a learned model's generalization result.

The three listening notes were fixed before training: cello A4 mf (first
string), trumpet A4 pp, and flute D5 pp. They belong to the validation
list. They illustrate three cases; they are not a corpus-wide listening
study. A flute phrase uses a prescribed eight-note score. The flute/cello
morph follows a prescribed embedding path at A4; neither is an
autonomously composed performance.

## Audio comparisons and numerical measurements

Preserve the raw WAVs written by the C++ applications. `vae_remap` already
applies edge fades and 99th-percentile normalization before PCM clipping;
the analysis must not describe that output as an unprocessed decoder
waveform. Analysis records rail-sample fractions and compares waveforms
after those application steps.

For a reference STFT round trip, use the same symmetric Hann and hop,
retain magnitude and phase, overlap-add with squared-window normalization,
and discard no extra information. This independent double-precision
NumPy reference is identified as such; it is not a new C++ execution.

Evaluate magnitude-spectrogram residual on the common-length signals,
using Hann 4,096 / hop 1,024, both at the original output level and with
a single least-squares gain fitted in the magnitude domain. The latter
is `sum((|X| - g |Y|)^2) / sum(|X|^2)`, where
`g = sum(|X||Y|) / sum(|Y|^2)`. It measures one spectral discrepancy,
not perceived quality or musical value. The gain-aligned metric cannot
establish loudness fidelity. Metrics use raw outputs, not listening copies.

Listening copies are mono PCM16 at 44.1 kHz. Match whole-clip RMS within
each original/reference/VAE/direct-fit/network group. Lower the common
target if necessary so every copy remains below -1 dBFS peak. This is
RMS matching, not a perceptual-loudness normalization. Save the exact gain
and final RMS/peak per file. The phrase and morph are separate gestures,
not level-matched reconstructions. No denoising or hand editing is applied.

The analysis writes the listening files, an offline player, figures, and
`v020/results.json`. The book quotes those recorded results. Updating a
figure requires updating its underlying record; successful CI does not
claim to have repeated these dataset training runs.

For distribution, `bundle.py` writes
`book/build/designing-musical-networks.zip` with the PDF at
`book/book.pdf` and the audio player at `notes/audio/index.html`.
Extract the whole archive and keep that structure so relative links
continue to work. To package a newly built PDF before copying it to the
tracked location, pass `--pdf book/build/book.pdf`.
