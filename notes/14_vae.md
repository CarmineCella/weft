# 14 — Variational autoencoders

The plain autoencoder learned to compress and reconstruct, and its
latent space was continuous enough that interpolating between two
*real* encoded points produced sensible blends. But it had a flaw we
flagged at the end of note 13: nothing forced the latent space into
any particular shape. The encoder scattered codes wherever was
convenient, leaving voids between the clusters. Interpolating between
known points works because the path stays near the data, but *sample*
a random latent code and you usually land in a void and decode
garbage. The space is good for reconstructing and interpolating known
points, useless for generating new ones.

The variational autoencoder fixes exactly this. The architecture is
almost the same; the changes are conceptual, and there are three of
them.

---

## 1. Encode a distribution, not a point

Instead of mapping an image to a single latent code, the VAE's encoder
maps it to a whole Gaussian distribution over latent space — a mean
`mu` and a variance `sigma^2`, one of each per latent dimension. The
idea is that an input shouldn't claim a single infinitely-precise
point; it should claim a little fuzzy region. During training we draw
a sample from that region and decode it, so the decoder is forced to
produce something sensible not just at one point but across the whole
neighbourhood. That is what makes nearby points in latent space decode
to similar images — continuity, by construction.

In code, the encoder simply emits `2 * LATENT` numbers instead of
`LATENT`. We slice the output: the first `LATENT` rows are `mu`, the
next `LATENT` rows are `logvar`.

Why `logvar` (the log of the variance) and not the variance directly?
A variance must be positive, which would mean constraining the
network's output. The logarithm is unconstrained — any real number is
a valid log-variance — so the network can output it freely, and we
recover a guaranteed-positive standard deviation with
`sigma = exp(0.5 * logvar)`.

---

## 2. The reparameterisation trick

We need to draw `z ~ N(mu, sigma^2)` and then backpropagate through
that draw to train `mu` and `sigma`. But sampling is random — there's
no derivative of "draw a random number" with respect to the
distribution's parameters. If the random node sits directly in the
path from parameters to loss, gradients can't pass.

The trick moves the randomness out of the path. Instead of sampling
`z` directly, draw a *standard* normal `eps ~ N(0, I)` — which depends
on no parameters at all — and compute

```
z = mu + sigma * eps
```

This `z` has exactly the distribution we wanted (`N(mu, sigma^2)`), but
now `eps` is just an input, like the image. The path from `mu` and
`sigma` to `z` is plain differentiable arithmetic:

```
dz/dmu    = 1
dz/dsigma = eps
```

So gradients flow through `mu` and `sigma` as usual, and the only
"random" thing, `eps`, is a leaf with nothing to learn. This is the
single idea that makes VAEs trainable by ordinary backprop.

In the example the trick is written out explicitly between the encoder
and decoder rather than hidden in a layer, because seeing the three
lines (`sigma = exp(0.5*logvar)`, draw `eps`, `z = mu + sigma*eps`) is
the whole point.

---

## 3. The KL divergence term

Encoding distributions buys continuity, but on its own it doesn't pull
the latent space toward any particular global shape — the encoder
could still place its little Gaussians anywhere. So we add a second
loss term that measures how far each encoded Gaussian `N(mu, sigma^2)`
is from the standard normal `N(0, I)`, and penalises the distance. The
measure is the Kullback–Leibler divergence, and for two Gaussians it
has a closed form:

```
KL = -0.5 * sum_d ( 1 + logvar_d - mu_d^2 - sigma_d^2 )
```

Minimising it pulls every `mu` toward 0 and every `sigma^2` toward 1 —
that is, it pulls all the per-image Gaussians toward the same unit
Gaussian centred at the origin. The encoded clusters get packed
together around the origin with no big gaps between them, because every
one of them is being tugged toward the same place.

That packing is what makes generation work. After training, the latent
space looks (globally) like a unit Gaussian, so if we *draw* a fresh
`z ~ N(0, I)` it lands where the training data actually mapped, and the
decoder turns it into a plausible new image.

The KL gradients we push back into the encoder (per element; the `1/B`
matches MSE's batch-averaging convention so the two loss terms compose
cleanly):

```
dKL/dmu     = (1/B) * mu
dKL/dlogvar = (1/B) * 0.5 * (sigma^2 - 1)
```

---

## 4. The total loss, and the balance

```
L = reconstruction (MSE)  +  beta * KL
```

These two terms pull in opposite directions, and that tension is the
heart of the VAE. Reconstruction wants the latent to carry as much
information about the input as possible (ideally a unique precise code
per image — back to the plain AE). KL wants every code to look like
generic `N(0, I)` noise (carrying *no* information). The useful
solution lives in between: carry enough information to reconstruct,
arranged in a smooth gap-free space.

`beta` sets the balance. With our MSE summed over the 784 pixels, the
reconstruction term has the right magnitude for `beta = 1` to be the
proper variational objective (the ELBO). Turn `beta` up and the latent
space becomes cleaner and more disentangled but reconstructions blur;
turn it down and reconstructions sharpen but the space develops the
same voids the plain AE had. Two failure modes worth naming:

- **Posterior collapse**: if KL dominates (or the decoder is powerful
  enough to do without the latent), the encoder gives up and outputs
  `mu = 0, sigma = 1` for everything — the KL term goes to zero and the
  latent carries nothing. You'd see KL crash toward 0 while
  reconstruction stays high. With a tight 2D bottleneck on MNIST this
  is unlikely; the decoder *needs* the latent.
- **Ignored prior**: if reconstruction dominates, KL stays large, the
  encoded Gaussians drift away from `N(0, I)`, and sampling degrades
  back toward AE-like voids.

A healthy run shows both terms settling to non-trivial, stable values —
recon decreasing then levelling, KL rising from ~0 then plateauing at
some modest positive number rather than collapsing or exploding.

---

## 5. The demos

**Reconstruction** uses `mu` directly (no sampling) — the deterministic
"best guess" code for each image. Output looks much like the plain AE:
recognisable, slightly blurry, denoised.

**Latent grid** is the demo the plain AE couldn't produce, and the one
that makes the whole thing click. With a 2D latent we sweep `z` over a
regular grid covering `[-2.5, 2.5]^2`, decode each grid point, and tile
the results into a single image. The result is the entire data
manifold laid out as one continuous sheet: each region of the plane
owns a kind of digit, and moving across the sheet morphs one into
another smoothly with no dead patches. That smooth, fully-populated
sheet is the visual proof that the KL term did its job.

(We space the grid linearly here for simplicity. The textbook version
spaces it by the inverse normal CDF of evenly-spaced probabilities, so
the grid samples the prior uniformly *by probability mass* — it makes
a slightly nicer manifold but needs a probit function we haven't
written.)

**Random samples** draw `z ~ N(0, I)` and decode. These are images the
encoder never produced from any real input — pure generation. Most are
clean digits; a few are blurry blends from samples that landed between
clusters.

---

## 6. What this sets up for audio

Nothing in the VAE machinery is image-specific. Swap the 784-pixel
input for a per-frame magnitude spectrum and the same encoder/decoder/
reparam/KL apparatus learns a latent space of *spectra* instead of
digits. Then:

- interpolating between two instruments' latent codes is a timbre morph
- sampling `z ~ N(0, I)` invents a new timbre that never existed
- decoding back to a spectrum, plus the input phase and an inverse
  STFT, turns it into audible sound

That's the morphing application. The remaining pieces are the audio
plumbing — STFT/iSTFT with overlap-add — not the model, which is done.
