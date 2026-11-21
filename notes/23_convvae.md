# 23 — A convolutional VAE for audio

The audio VAE of [chapter 19](19_audio_vae.md) works on individual STFT
frames. Each column of the magnitude spectrogram becomes an isolated
2049-dimensional vector that the dense encoder reduces to 32 latent
dimensions and the dense decoder reconstructs. That gives us a learned
dictionary of instantaneous spectra, but discards time. The
encoder never sees that the slice taken 50 ms into a violin attack
looks very different from the slice taken 500 ms into a violin sustain,
even though both are "violin." Anything that concerns how a sound
changes over time has to come from outside the model.

The natural fix is to encode short time-frequency patches instead of
single frames. A patch of 16 STFT frames × 64 frequency bins is a
small 2-D image whose vertical axis shows which frequencies are
present and whose horizontal axis shows how those frequencies move in
time. For a 2-D image we have almost everything we need: Conv2D,
MaxPool2D and ReLU4D from [chapter 22](22_convnets.md), plus one new
layer, `Upsample2D` (nearest-neighbour upsampling; its backward sums
the gradient over each block of copies). We stack them into a conv
encoder and a mirrored conv decoder with a small dense bottleneck in
between, and we have a VAE that models local time-frequency texture
as well as static timbre.

This chapter builds that ConvVAE and applies it to interpolating
between two timbres in latent space, sampling from the prior to
invent new timbres, and remapping arbitrary input audio onto the
manifold the model has learned. We also state where the result is
good and where it is not.


## 1. Patch shape

A first design choice is the patch geometry. With our existing STFT
settings (`FRAME_SIZE = 4096`, `HOP_SIZE = 2048`, 44.1 kHz sample rate)
each frame has 2049 magnitude bins spanning 0 to 22 kHz at ~10.77 Hz
per bin, and consecutive frames are 46 ms apart. We choose the
dimensions as follows:

- **Time dimension.** Sixteen frames is ~0.8 seconds. This is long
  enough to capture attack-to-sustain transitions and short enough
  that almost any TinySOL note produces at least one full patch.
  (Thirty-two frames = ~1.5 s would carry more context but would
  exclude all the shorter notes from training.)
- **Frequency dimension.** Sixty-four bins is small enough for a
  tractable conv stack, but only if we pick the right 64 bins.
  Choosing them causes most of the difficulty; see §3 below.

So the patch is `(1 channel, 64 freq, 16 time)`. The single channel
indicates that this is one magnitude image, not RGB.


## 2. Architecture

The encoder is a conv stack with three downsampling stages, followed
by a dense bottleneck that emits `μ` and `log σ²`. The decoder
mirrors it.

```
Encoder:
  Conv2D(1→16, 3×3) → ReLU → MaxPool(2)    (1,64,16) → (16,32,8)
  Conv2D(16→32, 3×3) → ReLU → MaxPool(2)             → (32,16,4)
  Conv2D(32→64, 3×3) → ReLU → MaxPool(2)             → (64, 8,2)
  Flatten                                            → 1024
  Dense(1024, 256) → ReLU → Dense(256, 2·LATENT)

z = μ + σ · ε,  ε ~ N(0, I)

Decoder:
  Dense(LATENT, 256) → ReLU → Dense(256, 1024) → ReLU
  Unflatten                                              → (64, 8, 2)
  Upsample(2) → Conv2D(64→32, 3×3) → ReLU                → (32,16, 4)
  Upsample(2) → Conv2D(32→16, 3×3) → ReLU                → (16,32, 8)
  Upsample(2) → Conv2D(16→1,  3×3)                       → (1,64,16)
```

**Linear output, no sigmoid.** Log-magnitudes are non-negative but
unbounded, not `[0, 1]` like pixel intensities. A sigmoid would
saturate them. The generation code clamps to `≥ 0` before
exponentiating back to linear magnitude.

**Reparameterisation inline.** As in [chapter 17](17_vae.md), the
encoder's last dense layer emits `2·LATENT` numbers; the first
half is `μ` and the second half is `log σ²`. We sample
`z = μ + σ · ε` with `ε ~ N(0, I)` and pass `z` to the decoder.
Gradients pass through `μ` and `σ` because the stochastic part (`ε`)
has no parameters.

**Four objects in code.** The training loop passes data through four
sub-networks (`enc_conv`, `enc_dense`, `dec_dense`, `dec_conv`)
because we have to convert between `Tensor4D` and `Matrix` at the
bottleneck:

```cpp
Tensor4D feat   = enc_conv.forward(Xb);
Matrix   flat   = flatten(feat);
Matrix   h      = enc_dense.forward(flat);          // [mu; logvar]
// ... sample z ...
Matrix   df     = dec_dense.forward(z);
Tensor4D dfeat  = unflatten(df, B, 64, 8, 2);
Tensor4D recon  = dec_conv.forward(dfeat);
```

Backward is the same chain in reverse. Each of the four sub-networks
saves to its own file, with suffixes `.enc_conv`, `.enc_dense`,
`.dec_dense`, `.dec_conv`. This is explicit and unambiguous. The
loading side rebuilds the architecture and reads the four files back.


## 3. The frequency-axis problem

Sixty-four bins cannot represent 2049 linear-frequency bins without
losing something. The choice is what to lose.

**Block-averaging destroys harmonic structure.** The naïve choice is
to average groups of 32 consecutive bins. A 440 Hz tone produces a
sharp peak at bin 41 in the original spectrum, with harmonics at bins
82, 123, 163, and so on. Averaging bins 32–63 mixes that one peak with
31 neighbouring valley bins, producing a low plateau value that no
longer looks like a peak. The VAE learns to reproduce those blurred
plateaus. When we upsample back to 2049 bins by repeating each output
value 32 times, we get a step-function spectrum with no sharp peaks
anywhere. Resynthesised, that sounds like broadband noise modulated
by the right energy envelope, because the envelope (overall energy
distribution) survives the averaging, but the harmonic peaks that
define pitched sound do not.

The reconstruction quality with block-averaging is bad enough that a
plain phase vocoder doing cross-synthesis on the raw 2049-bin
spectrogram sounds clearly better than the ConvVAE. That is a
problem: the conv VAE is supposed to add something a vocoder can't
do, but if the underlying audio quality is worse, none of the
latent-space arguments matter.

**Mel-frequency scaling solves the harmonic-smearing problem.** The
triangular mel filterbank concentrates resolution where the ear is
most sensitive: logarithmic spacing, narrow filters at low
frequencies, wide at high. With 64 mel bins covering 0–22 kHz at
44.1 kHz sample rate, roughly 35 of those bins sit below 4 kHz, where
most of the harmonic content is. A 440 Hz tone now contributes to
several adjacent mel bins whose triangles overlap the peak; the
harmonic structure survives the forward transform.

Going back from mel to linear uses the transpose of the filterbank
matrix as a pseudo-inverse. Each linear bin receives the sum, over
mel bins, of the mel bin's value weighted by the triangle that
originally fed it. This is lossy (`M^T M` is not the identity, because
the filters overlap), but it is the standard approximate inversion.
The spectrogram comes back blurred but with the peaks roughly where
they should be.

There is a stability problem that we found during training: mel
binning concentrates energy, so raw log-mel-magnitude values are about
9× larger than the corresponding log-mag values for the same audio.
Feeding those larger values into the encoder's randomly-initialised
final dense layer produces correspondingly larger `log σ²` outputs at
initialisation, and the `exp(log σ²)` term in the KL loss becomes very
large: we see KL values in the hundreds of thousands instead of the
expected single digits, and training oscillates strongly. The fix is
a constant scale factor: divide log-mel values by 8 on the way into
the network, multiply back by 8 on the way out. The constant is
stored in `MelTransform`'s state so training and inference always
agree.


## 4. Training

The loss is identical to that of the dense audio VAE: MSE between the
reconstructed patch and the input patch, plus β times the KL between
the encoder's `N(μ, σ²)` posterior and the standard normal prior, with
β = 1. Adam at learning rate 1e-3, batch size 32, twenty to thirty
epochs.

The data pipeline differs from the dense case. We read
the per-frame logmag cache (the same one [chapter 19](19_audio_vae.md)
uses), then for each contiguous run of frames belonging to one
source WAV, slide a 16-frame window with a hop of 8 frames and extract
patches. Each patch's 16 columns get mel-transformed from 2049 to 64
bins inside the loop, and the resulting `(1, 64, 16)` tile becomes one
training example. On TinySOL this gives about 24,000 patches, which
fits easily in memory.

In a healthy run the reconstruction loss decreases monotonically and
KL stays stable in single digits, with no oscillation. A representative
TinySOL run reaches recon ≈ 13 at epoch 1 and recon ≈ 7 by epoch 10,
with KL around 2 throughout. If KL spikes by factors of 100 or
recon bounces, the mel scale factor is wrong and the encoder is
producing pathological `log σ²` at init.


## 5. Applications

**`convae_generate interp`** encodes the middle 16-frame patch of
sound A to `z_A` and the middle patch of sound B to `z_B`. It sweeps
`z_k = (1-t) z_A + t z_B` for *t* from 0 to 1 across N output patches,
concatenates the N × 16 decoded mel-frames into a single
mel-spectrogram, mel-inverts to linear magnitudes, pairs them with
phases borrowed (and tiled) from A, and applies the inverse STFT. The
result is a single ~5–6 second audio clip whose magnitude evolves
smoothly from A's timbre to B's, with A's rhythmic phase character
throughout.

**`convae_generate sample`** draws `count` anchor latents from the
prior `N(0, I)` and moves linearly through them across N output
patches. A donor WAV supplies the phase. With `count = 1` it sustains
a single invented timbre; `count = 4` follows a trajectory through
four points of the learned timbre manifold.

**`convae_remap`** slides 16-frame patches across an arbitrary input
audio file with 50% overlap. It encodes each patch to its latent mean
(no sampling, because this application should be deterministic) and
decodes each back. It overlap-adds the reconstructions in the
mel-spectrogram domain, inverts mel to linear magnitude, pairs it with
the input's own STFT phase frame by frame, and applies the inverse
STFT. The output audio has the same duration as the input; the rhythm
and articulation are the input's, the timbre is whatever the VAE
projects.


## 6. Capabilities and limitations

A successful ConvVAE on TinySOL plus a remap pass over a song will
not sound like a clean phase vocoder doing cross-synthesis. The
vocoder works on the full 2049-bin spectrogram and leaves phase
untouched; the VAE has compressed each 1024-pixel patch through a
32-dimensional bottleneck and back. That compression is what gives us
a latent space to manipulate, but it costs audible fidelity, a cost
the vocoder does not have.

What the VAE does that a vocoder cannot is smooth manipulation in a
learned space. The vocoder has no latent code; pitch-shifting two
sounds toward each other or interpolating their formants are different
operations, each requiring its own DSP pipeline. The ConvVAE gives a
concrete manifold of timbres that can be traversed: the same
architecture gives us interp, sample, and remap with no per-operation
engineering.

The limitations of this build, from most to least damaging:

- **Mel inversion is lossy.** The transpose-as-pseudo-inverse blurs
  the spectrogram on the way back. Better mel inverters exist
  (learned vocoders like HiFi-GAN, iterative methods, Griffin-Lim
  starting from a good initialisation), but none are in the library.
- **Phase mismatch.** In remap, the output pairs the input's phase
  with the VAE's reconstructed magnitude. The phase encodes
  interference patterns specific to the original magnitude, so the
  pairing causes partial cancellations and an audible coloration.
  Griffin-Lim iteration would reconstruct phase from magnitude alone
  and remove this problem (~100 lines we don't have yet).
- **The latent bottleneck is severe.** 1024 pixels → 32 latents → 1024
  pixels is a 32× compression. Sustained tones survive the round trip
  well; rapid spectral changes (a piano attack, a percussive
  transient) blur into the surrounding sustain.

This chapter does not produce a finished audio system; for that one
would use a vocoder. What we have is the architectural pattern:
encoder as `ConvNetwork → Flatten → Network`, decoder as
`Network → Unflatten → ConvNetwork`, latent in the middle,
reparameterise to sample. With that pattern in place the same code
applies to any 2-D data with local spatial correlations, such as
spectrogram patches or image tiles, and in 1-D to raw waveforms.

Judged by ear, the audio from this model and from the dense VAE of
chapter 19 is poor, and the limitations above are structural: training
longer does not remove them. They all come from the same decision,
which is to make the network produce a spectrum and then turn that
spectrum back into a waveform.
[Chapter 24](24_differentiable_synthesis.md) keeps the goal and
changes the decision: the network there outputs the controls of a
synthesiser, and the synthesiser makes the sound.

![Spectrograms of the first 8 seconds of `cage.wav` and of its remap by the convolutional VAE, each scaled to its own maximum. The output has energy at all frequencies, and the harmonics of the input are no longer separate lines.](figures/23_remap.png)


## Further reading

The model combines the variational autoencoder of [@kingma2014vae] with
the convolutional layers of chapter 22. The checkerboard artefacts of
transposed convolutions, and the upsample-then-convolve remedy used in
our decoder, are analysed in [@odena2016deconvolution]. The mel scale is
[@stevens1937mel], and phase reconstruction from a magnitude
spectrogram is [@griffin1984signal].
