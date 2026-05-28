# 13 — Autoencoders

Everything so far has been *supervised*: we hand the network an input
and a label, and it learns the mapping. An autoencoder throws the
labels away. You give it an input and ask it to reproduce that same
input on the output — but force the signal through a narrow middle
layer first, so it can't just copy. To reconstruct well through a
bottleneck, the network has to discover the underlying structure of
the data and encode it compactly. That's representation learning, and
it's the doorway to generative models.

This note covers the two small primitives the autoencoder needs (MSE
loss and the Sigmoid activation), the autoencoder itself, and the
demo: reconstructing MNIST digits and morphing one digit into another
through the latent space. The demos are written out as BMP images via
a small `Bmp.h` writer (uncompressed 24-bit BMP — two headers and raw
pixels, the image-format equivalent of the WAV reader), so we can
actually look at the results instead of squinting at ASCII art.

---

## 1. MSE loss

For classification we used cross-entropy, which expects a probability
distribution. For an autoencoder reconstructing pixel intensities,
the natural objective is squared error: how far is each output pixel
from the corresponding input pixel?

```
L = (1 / 2N) * sum over examples, sum over features  (P - T)^2
```

`P` is the prediction, `T` the target, `N` the batch size. We sum over
features (the 784 pixels) and average over examples — same convention
as cross-entropy, where the per-example averaging factor `1/N` lives
in the loss so the layers never need to know the batch size.

The factor of `1/2` is a convenience: differentiating the square
brings down a 2 that cancels it, leaving a clean gradient:

```
dL/dP = (1/N) (P - T)
```

That's the whole loss. The gradient is literally the (averaged) error
itself — push each output toward its target in proportion to how
wrong it is. `tests/test_mse.cpp` checks the known value, the
gradient formula, and a numerical-gradient agreement.

One thing worth noticing: because we sum over 784 pixels, the loss
*value* looks large (tens, early in training) compared to the
classifier's cross-entropy (single digits). That's just the
summing-over-features convention — what matters is that it decreases.
Divide by `2 * 784` and take the square root to get a per-pixel RMS
error in the same [0, 1] units as the pixels.

---

## 2. Sigmoid activation

MNIST pixels live in [0, 1]. If the decoder's last layer were linear,
it could output any real number, including negatives and values above
1 — nonsense as pixel intensities. The logistic sigmoid squashes any
real input into (0, 1):

```
sigma(x) = 1 / (1 + e^-x)
```

so the reconstruction is always a valid intensity. Its derivative has
a famously tidy form:

```
sigma'(x) = sigma(x) (1 - sigma(x))
```

Since the forward pass already computed `Y = sigma(Z)`, the backward
pass reuses `Y` directly: `dZ = dY * Y * (1 - Y)`, element-wise. Like
ReLU, it's an element-wise activation, so the Jacobian is diagonal and
backward collapses to a Hadamard product. No parameters.

(We'll meet sigmoid again as the gate nonlinearity if we ever build
recurrent layers, and the same logistic shape is the binary case of
the softmax we already have.)

---

## 3. What an autoencoder is

The architecture is two halves:

```
encoder:  784 -> 128 -> ReLU -> 32 -> ReLU -> 16        (the bottleneck)
decoder:  16 -> 32 -> ReLU -> 128 -> ReLU -> 784 -> Sigmoid
```

The input (784 pixels) is squeezed down to a 16-number *latent code*,
then expanded back to 784 pixels. The loss is MSE between the output
and the original input. There are no labels anywhere in the training
objective — the data supervises itself.

The bottleneck is the whole point. 16 numbers cannot store 784
independent pixels, so the network is forced to find what's redundant.
MNIST digits are not random pixel soup; they live on a low-dimensional
manifold (strokes, curves, a handful of degrees of freedom). The
autoencoder discovers a coordinate system for that manifold. The
encoder maps an image to its coordinates; the decoder maps coordinates
back to an image.

Note the bottleneck layer itself is **linear** (no activation). We
want the latent code to be able to take any real value in any
direction; squashing it would distort the geometry we're about to
exploit for interpolation.

---

## 4. Why two Networks instead of one stack

A plain autoencoder is just a deep stack, so we *could* express it as
a single `Network` from 784 to 784. But building it as two separate
`Network`s — an encoder and a decoder — lets us call the halves
independently:

- `enc.forward(x)` → the latent code for an input
- `dec.forward(z)` → the image for a latent code

That separation is what makes the generative demos possible. With a
single stack you'd only ever get input-to-output; you couldn't reach
in and grab the latent code or feed the decoder a code of your own
choosing.

Training across two Networks needs no new library code — the latent
just flows between them:

```cpp
Matrix z      = enc.forward(Xb);        // 784 -> 16
Matrix recon  = dec.forward(z);         // 16  -> 784
float  L      = mse.forward(recon, Xb); // target = the input itself

Matrix dRecon = mse.backward();         // dL/d recon
Matrix dz     = dec.backward(dRecon);   // dL/d z, updates decoder grads
enc.backward(dz);                       // updates encoder grads (return ignored)

dec.update(opt);
enc.update(opt);
```

The chain rule runs straight through the boundary: the decoder's
backward returns the gradient with respect to its input (the latent
code), which is exactly the gradient the encoder's backward needs as
its incoming signal. One `Adam` instance optimises both networks —
it keys its per-parameter state on each weight matrix's address, so it
happily tracks parameters from any number of networks at once.

---

## 5. Demo 1 — reconstruction

After training, we run a few test digits through `enc` then `dec` and
write the originals and their reconstructions to a BMP image
(`ae_reconstructions.bmp`) using the `Bmp.h` writer: top row the
originals, bottom row the reconstructions, each digit scaled up so it's
comfortable to look at. A well-trained autoencoder produces a slightly
blurry but clearly recognisable copy. The blur is the bottleneck at
work — fine detail is exactly the high-dimensional information that 16
numbers can't retain, so the network keeps the broad strokes and drops
the rest. A pleasant side effect: because the noise in any single pixel
is unpredictable from the rest of the image, it can't survive the
bottleneck either, so reconstructions come out *denoised* relative to
their inputs.

---

## 6. Demo 2 — latent interpolation

This is the demo that shows the latent space is *meaningful* rather
than just a compression trick. Take two test digits, say a 3 and an 8.
Encode both to get codes `za` and `zb`. Now walk a straight line
between them in latent space and decode each point:

```
z(t) = (1 - t) za + t zb,   for t = 0, 0.2, 0.4, ... 1.0
```

Decoding each `z(t)` produces a sequence of images that smoothly morph
the 3 into the 8, written side by side into `ae_interpolation.bmp`.
The intermediate images are *not* averages of the two pictures (that
would just be a double-exposure ghost) — they are plausible digit-like
shapes that the decoder generates from points the encoder never
actually visited during training. The network has learned a continuous
space where "between a 3 and an 8" is a coherent place.

That continuity is the bridge to generation. If decoding arbitrary
points in latent space yields plausible outputs, then we can *sample*
new points and get new outputs — new digits, or in the audio version,
new timbres. The plain autoencoder gets us most of the way there, but
with a catch we'll fix next.

---

## 7. The catch, and what's next

A plain autoencoder's latent space has no enforced shape. The encoder
is free to scatter codes wherever is convenient — clusters here,
empty voids there, arbitrary scales on each axis. Interpolating
between two *real* encoded points usually works because the path
tends to stay near the data. But if you sample a random latent code
with no real image behind it, you'll often land in one of the voids
and the decoder produces garbage. The space is good for reconstructing
and interpolating known points, not for sampling fresh ones.

The **variational autoencoder** (next note) fixes this by forcing the
latent space into a known, gap-free shape — a unit Gaussian — so that
*any* sample from that distribution decodes to something plausible.
That requires two new ideas: encoding a distribution instead of a
point (the reparameterisation trick), and a second loss term that
pulls the latent distribution toward the standard normal (the KL
divergence). Once that's in place, generation is just: draw `z` from
a normal, decode, done.

And once generation works on MNIST, the same machinery points straight
at the audio goal: train the encoder/decoder on per-frame spectra
instead of pixels, and the latent space becomes a space of timbres to
interpolate through and sample from.
