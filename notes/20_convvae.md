# 20 — A convolutional VAE for audio

The audio VAE of [note 16](16_audio_vae.md) works on individual STFT
frames. Each column of the magnitude spectrogram becomes an isolated
2049-dimensional vector that the dense encoder squeezes to 32 latent
dimensions and the dense decoder reconstructs. That gives us a learned
dictionary of *instantaneous spectra*, but throws away time. The
encoder never sees that the slice taken 50 ms into a violin attack
looks very different from the slice taken 500 ms into a violin sustain,
even though both are "violin." Anything we want to do about how
a sound *changes* over time has to come from outside the model.

The natural fix is to encode short time-frequency *patches* instead of
single frames. A patch of 16 STFT frames × 64 frequency bins is a
small 2-D image whose vertical axis is *which frequencies are present*
and horizontal axis is *how those frequencies move in time*. Once it's
a 2-D image we have everything we need: Conv2D, MaxPool2D, ReLU4D,
Upsample2D, all from the convnet work of [note 19](19_convnets.md).
Stack them into a conv encoder and a mirrored conv decoder with a
small dense bottleneck in between, and we have a VAE that models
*local time-frequency texture*, not just static timbre.

This note builds that ConvVAE and three applications of it:
interpolating between two timbres in latent space, sampling from the
prior to invent new timbres, and remapping arbitrary input audio onto
the manifold the model has learned. We'll also be honest about where
the result is good and where it isn't.


## 1. Patch shape

A first design choice is the patch geometry. With our existing STFT
settings (`FRAME_SIZE = 4096`, `HOP_SIZE = 2048`, 44.1 kHz sample rate)
each frame has 2049 magnitude bins spanning 0 to 22 kHz at ~10.77 Hz
per bin, and consecutive frames are 46 ms apart. We need:

- **Time dimension.** Sixteen frames is ~0.8 seconds. Long enough to
  capture attack-to-sustain transitions, short enough that almost any
  TinySOL note produces at least one full patch. (Thirty-two frames
  = ~1.5 s would carry more context but would exclude all the shorter
  notes from training.)
- **Frequency dimension.** Sixty-four bins is small enough for a
  tractable conv stack but only *if we pick the right 64 bins*. That
  last clause is where most of the trouble lives; §3 below.

So the patch is `(1 channel, 64 freq, 16 time)`. The single channel
is the "this is one magnitude image, not RGB" placeholder.


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

A few small decisions worth flagging:

**Linear output, no sigmoid.** Log-magnitudes are non-negative but
unbounded, not `[0, 1]` like pixel intensities. A sigmoid would
saturate them. The generation code clamps to `≥ 0` before
exponentiating back to linear magnitude.

**Reparameterisation inline.** Same trick as [note 14](14_vae.md):
the encoder's last dense layer emits `2·LATENT` numbers; the first
half is `μ` and the second half is `log σ²`. We sample
`z = μ + σ · ε` with `ε ~ N(0, I)` and pass `z` to the decoder.
Gradients flow through `μ` and `σ` cleanly because the stochastic
part (`ε`) has no parameters.

**Four objects in code.** The training loop pipes data through *four*
sub-networks — `enc_conv`, `enc_dense`, `dec_dense`, `dec_conv` —
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
`.dec_dense`, `.dec_conv`. Explicit but unambiguous, and the
load-side rebuilds the architecture and reads the four files back.


## 3. The frequency-axis problem

Sixty-four bins cannot represent 2049 linear-frequency bins without
losing something. The question is *what*.

**The naïve choice — block-average groups of 32 consecutive bins —
destroys harmonic structure.** A 440 Hz tone produces a sharp peak
at bin 41 in the original spectrum, with harmonics at bins 82, 122,
163, and so on. Averaging bins 32–63 mixes that one peak with 31
neighbouring valley bins, producing a low plateau value that no longer
*looks* like a peak. The VAE learns to reproduce those blurry
plateaus. When we upsample back to 2049 bins by repeating each output
value 32 times, we get a step-function spectrum with no sharp peaks
anywhere. Resynthesised, that sounds like *broadband noise modulated
by the right energy envelope* — because the envelope (overall energy
distribution) survives the averaging, but the harmonic peaks that
define pitched sound do not.

The reconstruction quality with block-averaging is bad enough that a
plain phase vocoder doing cross-synthesis on the raw 2049-bin
spectrogram beats the ConvVAE handily. That's a problem: the whole
point of the conv VAE is that it should *add* something a vocoder can't
do, but if the underlying audio quality is worse, none of the
latent-space arguments matter.

**Mel-frequency scaling solves the harmonic-smearing problem.** The
triangular mel filterbank concentrates resolution where the ear cares
— logarithmic spacing, narrow filters at low frequencies, wide at high.
With 64 mel bins covering 0–22 kHz at 44.1 kHz sample rate, roughly
40 of those bins sit below 4 kHz where most of the harmonic action
lives. A 440 Hz tone now contributes to several adjacent mel bins
whose triangles overlap the peak; the harmonic structure survives the
forward transform.

Going back from mel to linear uses the *transpose* of the filterbank
matrix as a pseudo-inverse. Each linear bin receives the sum, over
mel bins, of the mel bin's value weighted by the triangle that
originally fed it. This is lossy — `M^T M` is not the identity, the
filters overlap — but it's the standard "good enough" inversion. The
spectrogram comes back blurry but with the peaks roughly where they
should be.

One stability gotcha learned the hard way: mel binning concentrates
energy, so raw log-mel-magnitude values are about 9× larger than the
corresponding log-mag values for the same audio. Feeding those larger
values into the encoder's randomly-initialised final dense layer
produces correspondingly larger `log σ²` outputs at initialisation, and
the `exp(log σ²)` term in the KL loss explodes — we see KL values in
the hundreds of thousands instead of the expected tens, and training
oscillates wildly. The fix is a constant scale factor: divide log-mel
values by 8 on the way into the network, multiply back by 8 on the way
out. The constant lives in `MelTransform`'s state so train and
inference always agree.


## 4. Training

The loss is identical to the dense audio VAE: MSE between the
reconstructed patch and the input patch, plus β times the KL between
the encoder's `N(μ, σ²)` posterior and the standard normal prior, with
β = 1. Adam at learning rate 1e-3, batch size 32, twenty to thirty
epochs.

What's different from the dense case is the data pipeline. We read
the per-frame logmag cache (the same one [note 16](16_audio_vae.md)
consumes), then for each contiguous run of frames belonging to one
source WAV, slide a 16-frame window with a hop of 8 frames and extract
patches. Each patch's 16 columns get mel-transformed from 2049 to 64
bins inside the loop, and the resulting `(1, 64, 16)` tile becomes one
training example. On TinySOL we typically get 50–80k patches, which
fits comfortably in memory.

Healthy training shows reconstruction loss monotonically decreasing,
KL holding stable in single digits, no oscillation. A representative
TinySOL run hits recon ≈ 13 at epoch 1 and recon ≈ 6 by epoch 10, with
KL hovering around 2–3 throughout. If KL spikes by factors of 100 or
recon bounces, the mel scale factor is wrong and the encoder is
producing pathological `log σ²` at init.


## 5. Three applications

**`convae_generate interp`** — encode the middle 16-frame patch of
sound A to `z_A`, the middle patch of sound B to `z_B`. Sweep
`z_k = (1-t) z_A + t z_B` for *t* from 0 to 1 across N output patches.
Concatenate the N × 16 decoded mel-frames into a single mel-spectrogram,
mel-invert to linear magnitudes, pair with phases borrowed (and tiled)
from A, inverse-STFT. The result is a single ~5–6 second audio clip
whose magnitude evolves smoothly from A's timbre to B's, with A's
rhythmic phase character throughout.

**`convae_generate sample`** — draw `count` anchor latents from the
prior `N(0, I)` and walk linearly through them across N output patches.
A donor WAV supplies the phase. With `count = 1` you sustain a single
invented timbre; `count = 4` walks a trajectory through four points
of the learned timbre manifold.

**`convae_remap`** — slide 16-frame patches across an arbitrary input
audio file with 50% overlap. Encode each patch to its latent *mean*
(no sampling — we want determinism for this application). Decode each
back. Overlap-add the reconstructions in the mel-spectrogram domain,
invert mel to linear magnitude, pair with the input's *own* STFT phase
frame-by-frame, inverse-STFT. The output audio has the same duration
as the input; the rhythm and articulation are the input's, the timbre
is whatever the VAE projects.


## 6. What this is and is not

A successful ConvVAE on TinySOL plus a remap pass over a song will
*not* sound like a clean phase vocoder doing cross-synthesis. The
vocoder works on the full 2049-bin spectrogram and leaves phase
untouched; the VAE has compressed each 1024-pixel patch through a
32-dimensional bottleneck and back. That compression is the whole
point — it's what gives us a latent space to manipulate — but it
costs audible fidelity that a vocoder doesn't have to pay.

What the VAE does that a vocoder cannot is *smooth manipulation in a
learned space*. The vocoder has no latent code; pitch-shifting two
sounds toward each other or interpolating their formants are different
operations, each requiring its own DSP pipeline. The ConvVAE makes a
manifold of timbres concrete and walkable: the same architecture
gives us interp, sample, and remap with no per-operation engineering.

Honest limitations from this build, in order of how much they hurt:

- **Mel inversion is lossy.** The transpose-as-pseudo-inverse blurs
  the spectrogram on the way back. Better mel inverters exist —
  learned vocoders like HiFi-GAN, iterative methods, Griffin-Lim
  starting from a good initialisation — but none are in the library.
- **Phase mismatch.** In remap, the output pairs the input's phase
  with the VAE's reconstructed magnitude. The phase encodes
  interference patterns specific to the *original* magnitude, so the
  pairing causes partial cancellations and an audible coloration.
  Griffin-Lim iteration would reconstruct phase from magnitude alone
  and close this gap (~100 lines we don't have yet).
- **The latent bottleneck is severe.** 1024 pixels → 32 latents → 1024
  pixels is a 32× compression. Sustained tones round-trip well;
  rapid spectral changes (a piano attack, a percussive transient)
  blur into the surrounding sustain.

The thing to take from this note isn't a finished audio system — for
that you'd reach for a vocoder. What we have is the *architectural
pattern*: encoder as `ConvNetwork → Flatten → Network`, decoder as
`Network → Unflatten → ConvNetwork`, latent in the middle,
reparameterise to sample. Once that pattern is in place the same
machinery applies to any 2-D data with local spatial correlations —
spectrogram patches today, image tiles tomorrow, raw waveforms in 1-D
the day after.

The next note picks up the other major architectural family we haven't
touched: **attention**. After that, the matching-pursuit + attention
generative audio system that ties Part II together.
