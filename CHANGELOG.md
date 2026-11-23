# Changelog

## Unreleased

- Added a concluding chapter linking the representations, measured
  results, listening examples, and three follow-up experiments.
- Reran MNIST with three declared seeds, the dense audio VAE, and the
  synthesis-control network using the v0.2.0 C++ sources. Recorded
  data hashes, complete file splits, commands, raw logs, and measurements.
- Added a Markdown listening prologue, 13 RMS-scaled audio examples,
  an offline player, and new experiment figures. Updated chapters
  12, 19, and 24 while identifying the remaining historical results.
- Added evidence/audio integrity checks and a portable PDF/listening
  archive to the book workflow.

## 0.2.0 — 2026-11-21

- Added stable logits-based SoftmaxCrossEntropy and moved classifier
  and transformer training examples to it. Confident wrong predictions
  retain a useful gradient even when softmax probabilities underflow.
- Added checkpoint version/configuration/scalar checks and reliable
  truncated-file rejection, including the legacy transformer reader.
  New headers distinguish same-shaped models with different activations,
  attention settings, or dropout probabilities.
- Added a master seed for model initialization and dropout. Training
  examples use deterministic initialization; MNIST accepts a seed and
  checkpoint path on the command line.
- Separated validation from final testing in the classifier examples.
  MNIST selects its saved model by validation loss. Group-aware splits
  now handle empty data and sparse file IDs.
- Added original frame offsets and preprocessing metadata to feature
  caches. ConvVAE patches no longer join frames across removed silence.
  Audio VAE programs share a checked preprocessing contract.
- Added numerical, persistence, timing, split, and reproducibility
  regressions, plus complete example smoke checks using synthetic data.
- Added Linux/macOS CI for GCC/Clang, portable/BLAS implementations,
  standalone headers, example pipelines, Markdown checks, and the PDF.
  Book compilation now propagates LaTeX failures.
- Removed the generic opening dedication and updated the book for the
  corrected APIs. Existing experimental figures/tables remain records
  of the earlier runs; they have not been presented as new benchmarks.

### Compatibility

Network/ConvNetwork version-one files and old headerless transformer
weights remain readable with the same architecture. Their old formats
cannot supply the stronger configuration guarantees of new files.
Classifier examples now end in logits; their saved architecture differs
from the earlier Softmax-ended examples.

Version-two feature caches can still be read, but do not carry the
offsets and preprocessing contract required by the audio VAE examples.
Regenerate them with extract_features.

Audio VAE applications now require matching preprocessing metadata in
their weight headers. Known-compatible legacy weights can be loaded
with the low-level Network/ConvNetwork API and re-saved with the
appropriate AudioModelConfig metadata after verifying the original
sample rate, window, FFT/hop, and mel settings. Otherwise retrain them.
No automatic conversion guesses those settings.
