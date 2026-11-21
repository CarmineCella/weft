# 19 — The audio VAE: training, serialization, and remapping

We have a working VAE (chapter 17) and WAV I/O (chapter 14). This
chapter adds the last audio component, `STFT.h`, a short-time
Fourier transform and its inverse by weighted overlap-add, whose
round trip reproduces a signal to the quantisation floor. It then
connects the STFT to the VAE: we train the VAE on the spectra of
orchestral sounds, save it, and use it to "remap" an arbitrary input
sound onto the manifold of orchestral timbres.

The chapter covers model serialization (so we train once), the
log-magnitude pipeline (which turns audio into VAE input and back
again), and the remap application.

## 1. Serialization: train once, apply many times

Training on the TinySOL frames is slow (a 2049-wide first layer over
hundreds of thousands of frames). We don't want to retrain every time
we remap a different song, so the network can now save and load its
weights. This is the same separation of slow training from fast
iteration that we already use for features.

The design is polymorphic, like `update(Optimizer&)`. `Layer` has two
new virtuals with no-op defaults:

```cpp
virtual void save_params(std::ostream&) const {}
virtual void load_params(std::istream&) {}
```

Activations inherit the no-ops (they have no parameters). `Dense`
overrides them to write and read its weight matrix and bias.
`Network::save` writes a small header (a `WNET` tag, a version, and the
layer count) and then asks each layer to serialise itself; `load` does
the reverse. `load` does not build the graph. It fills an
already-constructed network of the same architecture. Loading into
the wrong shape throws (the layer count and each matrix's dimensions are
checked). For this reason the application tools rebuild the identical
encoder/decoder before calling `load`.

The scheme needs no `dynamic_cast` and no central registry of layer
types. The convolutional layers of chapter 22 use the same two-method
scheme in their own parallel hierarchy.

## 2. The log-magnitude pipeline

Audio is a waveform; the VAE works on fixed-length vectors. The STFT
converts one to the other: it slices the signal into overlapping
frames, windows each frame, and takes its FFT. Each frame becomes a
complex spectrum, from which we take the **magnitude** (how much energy
at each frequency) and the **phase** (the alignment of each frequency).
We feed the VAE magnitude only. Phase is unstructured and known to be
hard for a network to model, so the standard approach is to model
magnitude and handle phase separately (see §4).

We feed **log-magnitude**, `log(1 + |X|)`, not raw magnitude.
Spectral magnitudes span a large dynamic range (a few loud harmonics
among many near-silent bins), and raw MSE would be dominated by the
loud bins. The logarithm compresses that range into roughly `[0, 6]`,
so the network's capacity is used across the whole spectrum. This is
the same `logmag` feature the classifier used, so the existing
`extract_features ... logmag per_frame` produces the training set, and
the remap computes the identical quantity from its own STFT (same Hann
window, same FFT, same `log(1 + mag)`). There is no mismatch between
training and inference.

The VAE is the MNIST one with bigger layers and a 32-D latent. We
don't need a 2-D latent here, because there is no manifold picture to
draw and only quality matters. One change: the decoder's output layer
is linear, not sigmoid. Pixels were bounded to `[0, 1]`; log-magnitudes
are non-negative and unbounded, so no squashing is needed. The apps
clamp the decoder output to `>= 0` and invert with `expm1` to recover
magnitude.

## 3. Training (`train_audio_vae`)

`train_audio_vae` reads the per-frame logmag cache, ignores the labels
(this is unsupervised), and runs the VAE training loop from chapter 17:
the encoder emits `mu` and `logvar`, we reparameterise
`z = mu + sigma*eps`, decode, compute the loss `= MSE + beta*KL`, and
backpropagate the reconstruction and KL gradients into the encoder. At
the end it saves `<prefix>.enc` and `<prefix>.dec`.

This is also the first training run heavy enough to need attention to
performance: the frames can be subsampled (an optional argument, off
by default; adjacent STFT frames are highly redundant) and the model
is checkpointed periodically, so an interrupted long run does not lose
everything. These techniques, and the parallelised matmul they rely
on, are the subject of chapter 20.

On the balance of the loss terms: because MSE is summed over 2049
bins, the reconstruction term is two to three times the KL term at
`beta = 1`, so the latent isn't pulled tightly onto the unit Gaussian.
This is the "ignored prior" regime from chapter 17. It is fine for the
remap (which encodes real frames and uses `mu`, never sampling the
prior), but for clean sampling and interpolation we need a larger
`beta`. It is a constant in the file.

## 4. Remapping (`vae_remap`): projection onto the manifold

This is the application that the audio chapters lead to. Take any
input (a song, a voice) and pass each of its magnitude frames through
the VAE. The decoder has only ever learned to produce TinySOL-like
spectra, so an out-of-distribution frame (a drum hit, a vocal formant)
is reconstructed as its nearest orchestral interpretation. The result
is the input's rhythm and structure rendered in orchestral timbre.
Conceptually, this autoencodes out-of-distribution input through a
decoder constrained to one manifold: it projects the song onto the
space of orchestral sounds.

The pipeline:

1. STFT the input → complex frames.
2. For every frame, compute `log(1 + magnitude)` and store the phase.
3. Stack all frames into one `2049 x n_frames` matrix and run a single
   batched `encode -> mu -> decode`. (One large matmul is faster than
   thousands of one-column forward passes, and the `Network` is already
   built for batches, columns = examples.)
4. For each frame, invert the decoder's log-magnitude (`expm1`, clamp
   `>= 0`) and recombine it with that frame's **original phase**.
5. iSTFT, a short fade at both ends, and normalisation to the 99th
   percentile of the absolute sample values.

Step 4 reuses the input's phase. The VAE changes the magnitude (which
frequencies are present), and we keep the input's phase (when they
occur and how they are aligned). We do not have to estimate a phase,
so there are no phase-reconstruction artifacts. This is why remapping
is more robust than pure generation (section 6).

## 5. What to expect from the remap

This is a dense network on magnitude frames with reused phase. Expect
the input rendered with orchestral timbre, recognisable and of
demonstration quality, not the quality of a commercial timbre-transfer
plugin. Each frame is treated independently (the model has no notion
of time), so sustained tones fare better than sharp transients. The
orchestral colour comes from the decoder being unable to represent
anything but TinySOL spectra.

![Spectrograms of the first 8 seconds of `cage.wav` and of its remap by the dense VAE. Each panel is scaled to its own maximum. The output follows the timing of the input. Its spectrum is denser than the input's, and the pauses are not silent.](figures/19_remap.png)

## 6. Generating (`vae_generate`): interpolation and sampling

The remap could reuse the input's phase. The two generative
applications produce a magnitude with no obvious phase, so each has to
borrow one. Both avoid iterative phase estimation
(Griffin-Lim [@griffin1984signal]), trading some fidelity for
simplicity.

**Interpolation** morphs between two real sounds. Encode both to latent
means, then for each output frame blend the codes,
`z = (1-t)*z_A + t*z_B`, with `t` going linearly from 0 at the first
output frame to 1 at the last, and decode. The phase is reused from
sound A, frame by frame. Because interpolation has a real source to
borrow phase from, it sounds nearly as clean as the remap on sustained
material. Coherence degrades as `t` approaches 1 and A's phase no
longer matches B's magnitude, and it is worst on transients.

**Sampling** is the purely generative case: draw `count` anchor latents
`z ~ N(0, I)` and move linearly through them across the donor's
duration. Frame 0 lands on anchor 0, the last frame lands on anchor
`count-1`, and interior frames interpolate between adjacent anchors, so
one output passes through several invented timbres. This also shows
the continuity of the latent space, a characteristic property of the
VAE: nearby latents decode to similar sounds. The cost comes from
phase. There is no source at all, so we borrow the phase of the donor
sound frame by frame. This is the crudest path: the donor's phase only
fits the frequency bins where the donor had energy, and the sampled
magnitude generally peaks elsewhere, so the mismatch produces
artifacts. It is the case a real phase-reconstruction method
(Griffin-Lim, or a neural vocoder) would improve most. We include it
because hearing a timbre drawn from the prior, however rough, is the
reason to have a VAE rather than a plain autoencoder.

The phase problem is not a limitation of the VAE. Sampling produces a
complete, valid magnitude spectrum. The difficulty is that our chosen
representation (magnitude only) discards phase, so turning that
spectrum back into a waveform needs phase from somewhere. On images
there is no such cost: a sampled image is the finished output. The
cost comes from the audio representation, not from the generative
model. This is the historical motivation for the neural-vocoder line
of work; here we borrow phase and go no further.

## 7. Limitations

A weak `beta` keeps the latent loosely organised. This matters little
for the remap but does matter for sampling and interpolation, where a
better organised latent would morph and sample more smoothly. The
deepest limitation is that the per-frame model has no notion of time:
timbre that should evolve coherently is reconstructed one independent
frame at a time. Both limitations point toward a convolutional VAE
that sees the 2D time-frequency structure of a spectrogram
([chapter 23](23_convvae.md)), and toward models of sequences
(chapters 25 to 29).

The audio quality is low. The remap, the interpolation and the
sampling all run, and they show what a latent space can be used for,
but a small network trained with squared error produces blurred
spectra, and the phase is borrowed from another sound. Chapter 23
improves the model; it does not change that.
[Chapter 24](24_differentiable_synthesis.md) gets clean audio by a
different route, with a network that controls a synthesiser instead of
producing spectra.


## Further reading

The model is the variational autoencoder of [@kingma2014vae] applied to
spectrogram frames of the samples described in [@cella2020orchidea].
Estimating a phase for a given magnitude spectrogram, which we avoid by
borrowing one, is the problem solved iteratively by
[@griffin1984signal]. The learned alternative, a network that produces
the waveform directly, begins with [@vandenoord2016wavenet].
