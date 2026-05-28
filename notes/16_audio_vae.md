# 16 — The audio VAE: training, serialization, and remapping

We have a working VAE (note 14) and the audio plumbing — WAV I/O and a
transparent STFT/iSTFT (the round-trip reproduces a signal to the
quantisation floor). This note connects them: train the VAE on the
spectra of orchestral sounds, save it, and use it to "remap" an
arbitrary input sound onto the manifold of orchestral timbres.

Three pieces: model serialization (so we train once), the
log-magnitude pipeline (turning audio into something the VAE eats and
back again), and the remap application itself.

## 1. Serialization — train once, apply many times

Training on the SOL frames is slow (a 2049-wide first layer over
hundreds of thousands of frames). We don't want to retrain every time
we remap a different song, so the network can now save and load its
weights — exactly the train-once / iterate-fast split we already use
for features.

The design is polymorphic, mirroring how `update(Optimizer&)` works.
`Layer` gained two virtuals with no-op defaults:

```cpp
virtual void save_params(std::ostream&) const {}
virtual void load_params(std::istream&) {}
```

Activations inherit the no-ops (they have no parameters); `Dense`
overrides them to write and read its weight matrix and bias.
`Network::save` writes a small header (a `WNET` tag, a version, and the
layer count) and then asks each layer to serialise itself; `load` does
the reverse. Crucially, `load` does **not** build the graph — it fills
an already-constructed network of the *same architecture*. Loading into
the wrong shape throws (the layer count and each matrix's dimensions are
checked). That's why the application tools rebuild the identical
encoder/decoder before calling `load`.

No `dynamic_cast`, no central registry of layer types: a future Conv
layer just implements the two methods and serialises for free.

## 2. The log-magnitude pipeline

Audio is a waveform; the VAE works on fixed-length vectors. The STFT
bridges them: slice into overlapping frames, window, FFT. Each frame
becomes a complex spectrum, from which we take the **magnitude** (how
much energy at each frequency) and the **phase** (the alignment of each
frequency). We feed the VAE magnitude only — phase is unstructured and
notoriously hard for a network to model, so the standard move is to
model magnitude and handle phase separately (see §4).

We don't feed raw magnitude, but **log-magnitude**, `log(1 + |X|)`.
Spectral magnitudes span a huge dynamic range — a few loud harmonics
over a sea of near-silent bins — and raw MSE would obsess over the loud
bins. The logarithm compresses that range into roughly `[0, 6]`, so the
network spends its capacity sensibly across the spectrum. This is the
same `logmag` feature the classifier used, so the existing
`extract_features ... logmag per_frame` produces exactly the training
set, and the remap computes the identical quantity from its own STFT
(same Hann window, same FFT, same `log(1 + mag)`) — no train/inference
mismatch.

The VAE itself is the MNIST one with bigger layers and a 32-D latent
(we don't need a 2-D latent here — there's no manifold picture to draw,
just quality). One change: the decoder's output layer is **linear**, not
sigmoid. Pixels were bounded to `[0, 1]`; log-magnitudes are
non-negative and unbounded, so there's nothing to squash. The apps
clamp the decoder output to `>= 0` and invert with `expm1` to recover
magnitude.

## 3. Training (`train_audio_vae`)

`train_audio_vae` reads the per-frame logmag cache, ignores the labels
(this is unsupervised), and runs the exact VAE training loop from
note 14 — encoder emits `mu` and `logvar`, reparameterise
`z = mu + sigma*eps`, decode, loss `= MSE + beta*KL`, backprop the
reconstruction and KL gradients into the encoder. At the end it saves
`<prefix>.enc` and `<prefix>.dec`.

A balance note: because MSE is summed over 2049 bins, the
reconstruction term dwarfs the KL at `beta = 1`, so the latent isn't
pulled tightly onto the unit Gaussian. That's the "ignored prior" regime
from note 14 — perfectly fine for the remap (which encodes real frames
and uses `mu`, never sampling the prior), but for clean *sampling* and
*interpolation* we'll want a larger `beta`. It's a constant in the file.

## 4. Remapping (`vae_remap`) — projection onto the manifold

This is the application the whole audio arc was built for. Take any
input — a song, a voice — and pass each of its magnitude frames through
the VAE. The decoder has only ever learned to produce SOL-like spectra,
so an out-of-distribution frame (a drum hit, a vocal formant) is
reconstructed as its *nearest orchestral interpretation*. The result is
the input's rhythm and structure rendered in orchestral timbre.
Conceptually it's autoencoding out-of-distribution input through a
decoder constrained to one manifold — projecting the song onto the space
of orchestral sounds.

The pipeline:

1. STFT the input → complex frames.
2. For every frame, compute `log(1 + magnitude)` and stash the phase.
3. Stack all frames into one `2049 x n_frames` matrix and run a single
   batched `encode -> mu -> decode`. (One big matmul beats thousands of
   one-column forward passes — and the `Network` is already built for
   batches, columns = examples.)
4. For each frame, invert the decoder's log-magnitude (`expm1`, clamp
   `>= 0`) and recombine it with that frame's **original phase**.
5. iSTFT and peak-normalise.

The phase reuse in step 4 is the key to why this application is robust.
The VAE changed the magnitude — the *what frequencies* — but we keep the
input's phase — the *when and how aligned*. We're not inventing phase
from nothing, so there are no phase-reconstruction artifacts. That's why
remapping sounds clean while pure generation (next) is harder.

## 5. What to expect from the remap

This is a dense network on magnitude frames with reused phase: expect
"input rendered through the orchestra," recognisable and demo-quality,
not a commercial timbre-transfer plugin. Each frame is treated
independently (the model has no sense of time), so sustained tones fare
better than sharp transients, and the orchestral "colour" comes from the
decoder being unable to represent anything but SOL spectra.

## 6. Generating (`vae_generate`): interpolation and sampling

The remap had the input's phase to lean on. The two generative
applications produce a magnitude with no obvious phase, so each has to
borrow one. Both avoid iterative phase estimation (Griffin-Lim), trading
some fidelity for simplicity.

**Interpolation** morphs between two real sounds. Encode both to latent
means, then for each output frame blend the codes —
`z = (1-t)*z_A + t*z_B` — and decode. With `t` swept 0->1 over the
duration the timbre travels from A to B; with a fixed `t` it holds a
static hybrid ("halfway between these two instruments"). The phase is
reused from sound A, frame by frame. Because interpolation has a genuine
source to borrow phase from, it sounds nearly as clean as the remap on
sustained material; coherence degrades as `t` approaches 1 and A's phase
no longer matches B's magnitude, worst on transients.

**Sampling** is the pure generative case: draw `z ~ N(0, I)`, decode to a
single invented magnitude spectrum, and sustain it. Here there is no
source at all, so we borrow the phase of a *donor* sound frame by frame.
This is the crudest path, and honestly so: the donor's phase only "fits"
the frequency bins where the donor had energy, and the sampled magnitude
generally peaks elsewhere, so the mismatch produces artifacts. It is
exactly the case a real phase-reconstruction method (Griffin-Lim, or a
neural vocoder) would most improve. We include it because hearing a
timbre drawn straight from the prior — however rough — is the point of
having a VAE rather than a plain autoencoder.

A subtlety worth stating plainly: the phase problem is **not** a
limitation of the VAE. Sampling produces a complete, valid magnitude
spectrum; the difficulty is purely that our chosen representation
(magnitude only) discards phase, so turning that spectrum back into a
waveform needs phase from somewhere. On images there is no such tax — a
sampled image is the finished output. It is the audio representation, not
the generative model, that charges the toll. That realisation is the
historical motivation for the entire neural-vocoder line of work; here we
simply borrow phase and move on.

## 7. Bigger picture

A weak `beta` keeps the latent loosely organised, which barely matters
for the remap but does for sampling and interpolation, where a tidier
latent would morph and sample more smoothly. And the per-frame model's
blindness to time is the deepest limitation: timbre that should evolve
coherently is reconstructed frame by independent frame. Both point the
same way -- toward a convolutional VAE that sees the 2D time-frequency
structure of a spectrogram, and a sequence model on top of its latents.
Those are the next milestones.

