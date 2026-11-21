# Listening files

Open [the offline player](index.html), then compare the original recording
with each reconstruction. These files accompany the book's
[listening prologue](../00_listening.md). The player needs no server.

## Fixed listening cases

| Group | TinySOL source path | Split |
| --- | --- | --- |
| Cello A4 mf | Strings/Vc/ordinario/Vc-ord-A4-mf-1c.wav | Validation |
| Trumpet A4 pp | Brass/TpC/ordinario/TpC-ord-A4-pp.wav | Validation |
| Flute D5 pp | Winds/Fl/ordinario/Fl-ord-D5-pp.wav | Validation |

All three files were chosen before the new models were trained. Other
recordings of these instruments and pitches can be in training.
The full split is recorded in
[the corpus manifest](../../book/experiments/v020/corpus.json).

The numbered examples in the prologue are:

1. [Cello recording](v020/cello_original.wav).
2. [STFT round trip](v020/cello_stft.wav), an independent NumPy reference.
3. [Dense VAE](v020/cello_vae.wav), latent mean decoded with the input phase.
4. [Direct fit](v020/cello_fit.wav), controls optimized for this target.
5. [Network-controlled synthesiser](v020/cello_network.wav), with pitch and
   loudness supplied from the target.
6. [Flute phrase](v020/gesture_phrase.wav), using the example's prescribed
   eight-note score.
7. [Flute/cello morph](v020/gesture_morph.wav), following a prescribed path
   between the learned instrument vectors at A4.

The player also includes the trumpet and flute cases. The phrase and
morph use explicit musical controls; their notes and trajectory were not
composed by the network.

## Processing

Files are mono PCM16 at 44.1 kHz. Within each reconstruction group the
whole-clip RMS is matched to a common level, reduced when needed to keep
all peaks below -1 dBFS. This is RMS matching, not perceptual loudness
matching. The separate gesture clips are scaled independently. There is
no additional denoising, editing, or repair.

The raw `vae_remap` output already has the example's edge fades and
99th-percentile normalization, including any clipping incurred when
writing PCM16. The listening copy retains the result of that processing.
The raw synthesis outputs have no added pitch or noise correction.

Exact gains, durations, RMS/peak values, file hashes, raw-output
measurements, and training results are in
[results.json](../../book/experiments/v020/results.json).
The [protocol and reproduction commands](../../book/experiments/README.md)
explain how to regenerate the raw outputs and listening copies.

## Source credit

The source recordings come from TinySOL, derived from Studio On Line,
recorded at Ircam. The dataset creators are Carmine-Emanuele Cella,
Daniele Ghisi, Vincent Lostanlen, Fabien Lévy, Joshua Fineberg, and
Yan Maresz.

The data used here are the 1,532-file
[Orchidea TinySOL_2020 distribution](https://www.orch-idea.org/datasets/).
See the [TinySOL dataset record](https://zenodo.org/records/3685367) for
credits and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
terms. The original listening copies are level-scaled; STFT, VAE, and
synthesiser clips are processed versions or generated outputs. The exact
source identities and changes are documented above.
