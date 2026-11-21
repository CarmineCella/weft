# 17 — Variational autoencoders

The plain autoencoder learned to compress and reconstruct, and its
latent space was continuous enough that interpolating between two
real encoded points produced sensible blends. It had the flaw
described at the end of chapter 16: nothing forced the latent space
into any particular shape. The encoder placed codes wherever was
convenient, leaving voids between the clusters. Interpolating between
known points works because the path stays near the data, but a
randomly sampled latent code usually lands in a void and decodes to
meaningless output. The space is good for reconstructing and
interpolating known points, and useless for generating new ones.

The variational autoencoder fixes this. The architecture is almost
the same. The changes are conceptual.

---

## 1. Encode a distribution, not a point

Instead of mapping an image to a single latent code, the VAE's encoder
maps it to a Gaussian distribution over latent space: a mean `mu` and
a variance `sigma^2`, one of each per latent dimension. An input then
occupies a small region of latent space instead of a single point.
During training we draw a sample from that region and decode it, so
the decoder is forced to produce something sensible across the whole
neighbourhood and not only at one point. This makes nearby points in
latent space decode to similar images, so continuity holds by
construction.

In code, the encoder emits `2 * LATENT` numbers instead of `LATENT`.
We slice the output: the first `LATENT` rows are `mu`, the next
`LATENT` rows are `logvar`.

We use `logvar` (the log of the variance) and not the variance itself
because a variance must be positive, which would mean constraining the
network's output. The logarithm is unconstrained (any real number is
a valid log-variance), so the network can output it freely, and we
recover a standard deviation that is guaranteed positive with
`sigma = exp(0.5 * logvar)`.

---

## 2. The reparameterisation trick

We need to draw `z ~ N(mu, sigma^2)` and then backpropagate through
that draw to train `mu` and `sigma`. But sampling is random, and
drawing a random number has no derivative with respect to the
distribution's parameters. If the random node sits directly in the
path from parameters to loss, gradients can't pass.

The trick moves the randomness out of the path. Instead of sampling
`z` directly, draw a standard normal `eps ~ N(0, I)`, which depends on
no parameters, and compute

```
z = mu + sigma * eps
```

This `z` has the distribution we wanted (`N(mu, sigma^2)`), but now
`eps` is an input, like the image. The path from `mu` and `sigma` to
`z` is plain differentiable arithmetic:

```
dz/dmu    = 1
dz/dsigma = eps
```

So gradients flow through `mu` and `sigma` as usual, and the only
random quantity, `eps`, is a leaf with nothing to learn. This is what
makes VAEs trainable by ordinary backprop.

In the example the trick is written out explicitly between the encoder
and decoder and not hidden in a layer, so that the three lines
(`sigma = exp(0.5*logvar)`, draw `eps`, `z = mu + sigma*eps`) are
visible.

---

## 3. The KL divergence term

Encoding distributions gives continuity, but on its own it doesn't
pull the latent space toward any particular global shape: the encoder
could still place its Gaussians anywhere. So we add a second loss
term that measures how far each encoded Gaussian `N(mu, sigma^2)` is
from the standard normal `N(0, I)`, and penalises the distance. The
measure is the Kullback–Leibler divergence, and for two Gaussians it
has a closed form:

```
KL = -0.5 * sum_d ( 1 + logvar_d - mu_d^2 - sigma_d^2 )
```

Minimising it pulls every `mu` toward 0 and every `sigma^2` toward 1.
That is, it pulls all the per-image Gaussians toward the same unit
Gaussian centred at the origin. The encoded clusters are packed
together around the origin with no large gaps between them, because
all of them are pulled toward the same place.

This packing is what makes generation work. After training, the latent
space looks (globally) like a unit Gaussian, so a fresh draw
`z ~ N(0, I)` lands where the training data mapped, and the decoder
turns it into a plausible new image.

The KL gradients we push back into the encoder are (per element; the
`1/B` matches MSE's batch-averaging convention so that the two loss
terms combine consistently):

```
dKL/dmu     = (1/B) * mu
dKL/dlogvar = (1/B) * 0.5 * (sigma^2 - 1)
```

---

## 4. The total loss, and the balance

```
L = reconstruction (MSE)  +  beta * KL
```

These two terms pull in opposite directions, and the VAE depends on
the balance between them. Reconstruction favours a latent that carries
as much information about the input as possible (ideally a unique
precise code per image, as in the plain AE). KL favours codes that all
look like generic `N(0, I)` noise (carrying no information). The
useful solution lies in between: enough information to reconstruct,
arranged in a smooth space without gaps.

`beta` sets the balance. With our MSE summed over the 784 pixels, the
reconstruction term has the right magnitude for `beta = 1` to be the
proper variational objective (the ELBO). Increasing `beta` makes the
latent space cleaner and more disentangled but blurs reconstructions.
Decreasing it sharpens reconstructions but the space develops the
same voids the plain AE had. The failure modes are:

- **Posterior collapse**: if KL dominates (or the decoder is powerful
  enough to do without the latent), the encoder outputs
  `mu = 0, sigma = 1` for everything. The KL term goes to zero and the
  latent carries nothing. KL falls toward 0 while reconstruction stays
  high. With a tight 2D bottleneck on MNIST this is unlikely, because
  the decoder needs the latent.
- **Ignored prior**: if reconstruction dominates, KL stays large, the
  encoded Gaussians drift away from `N(0, I)`, and sampling degrades
  back toward AE-like voids.

A good run shows both terms settling to non-trivial, stable values:
recon decreases then levels off, and KL rises from ~0 then plateaus at
a modest positive number, without collapsing or growing without bound.

---

## 5. Latent dimension: quality vs. the manifold picture

This example has a specific trade-off. A 2D latent lets us draw the
manifold grid (below), which is the clearest demonstration that the
space is continuous and has no gaps. But two numbers cannot describe a
specific handwritten digit, with its slant, stroke width and loops, so
2D reconstructions are necessarily a blurry class-average. A 2-float
bottleneck does not have the capacity for 60,000 varied images. A 2D
run lands around 0.20 RMS error per pixel, versus ~0.12 for the
16-dimensional plain autoencoder in chapter 16.

The example defaults to LATENT = 2, so that the manifold picture can
be drawn. Set it to 16 for sharper reconstructions and more varied
samples (the grid is then skipped). Raising the dimension has the
largest effect on quality; more epochs help only marginally once recon
stops falling. (A second, smaller effect: MSE itself blurs generative
image output, because the MSE-optimal guess under uncertainty is the
average of all plausible outputs. Per-pixel binary cross-entropy
sharpens MNIST, and `beta < 1` trades latent regularity for
reconstruction sharpness.)

## 6. The demos

**Reconstruction** uses `mu` directly (no sampling), the deterministic
best-guess code for each image. At 2D the reconstructions are the
blurry averages described above; at 16D they are much sharper.

**Latent grid** (only drawn when `LATENT == 2`) is the demo the plain
AE couldn't produce. We sweep `z` over a regular grid covering
`[-2.5, 2.5]^2`, decode each grid point, and tile the results into a
single image. It shows the data manifold as one continuous sheet, in
which each region holds a kind of digit and changes smoothly into its
neighbours with no dead patches. That the sheet is fully populated
shows that the KL term had its intended effect.

(We space the grid linearly for simplicity. The textbook version spaces
it by the inverse normal CDF of evenly-spaced probabilities, which
samples the prior uniformly by probability mass. That gives a slightly
better manifold picture, but it needs a probit function we haven't
written.)

**Latent interpolation** works in any dimension: encode two digits to
their means, walk the straight line between the two codes, and decode
each step. One digit morphs into another through latent space. This is
the same idea as the AE's interpolation, now in a space that has been
regularised to be smooth everywhere.

**Random samples** draw `z ~ N(0, I)` and decode. These are images the
encoder never produced from any real input, so they are generated and
not reconstructed. Most are readable digits; a few are blends from
samples landing between clusters.

---

## 7. Application to audio

Nothing in the VAE is image-specific. Replace the 784-pixel input with
a per-frame magnitude spectrum and the same encoder, decoder,
reparameterisation and KL term learn a latent space of spectra instead
of digits. Then:

- interpolating between two instruments' latent codes is a timbre morph
- sampling `z ~ N(0, I)` produces a new timbre that never existed
- decoding back to a spectrum, plus the input phase and an inverse
  STFT, turns it into audible sound

This is the morphing application. The model is complete. The remaining
piece is the audio processing, STFT/iSTFT with overlap-add, added in
[chapter 19](19_audio_vae.md).


## Further reading

The variational autoencoder and the reparameterisation trick are
[@kingma2014vae]. The effect of weighting the KL term, the `beta` of
this chapter, is studied in [@higgins2017betavae].
