# 16 — Autoencoders

Everything so far has been *supervised*: we give the network an input
and a label, and it learns the mapping. An autoencoder uses no labels.
It receives an input and is asked to reproduce that same input at the
output, but the signal must first pass through a narrow middle layer,
so the network cannot copy it. To reconstruct well through this
bottleneck, the network has to encode the underlying structure of the
data compactly. This is representation learning, and generative models
build on it.

This chapter covers the small primitives the autoencoder needs (MSE
loss and the Sigmoid activation), the autoencoder itself, and the
demo: reconstructing MNIST digits and morphing one digit into another
through the latent space. The demos are written out as BMP images by
a small `Bmp.h` writer (uncompressed 24-bit BMP: two headers and raw
pixels, the image counterpart of the WAV reader), so we can look at
the results as images instead of ASCII art.

---

## 1. MSE loss

For classification we used cross-entropy, which expects a probability
distribution. For an autoencoder reconstructing pixel intensities,
the natural objective is squared error, which measures how far each
output pixel is from the corresponding input pixel.

```
L = (1 / 2N) * sum over examples, sum over features  (P - T)^2
```

`P` is the prediction, `T` the target, `N` the batch size. We sum over
features (the 784 pixels) and average over examples. This is the same
convention as cross-entropy, where the per-example averaging factor
`1/N` lives in the loss so the layers never need to know the batch
size.

The factor of `1/2` is a convenience: differentiating the square
brings down a 2 that cancels it, leaving the gradient

```
dL/dP = (1/N) (P - T)
```

The gradient is the (averaged) error: each output is pushed toward its
target in proportion to its error. `tests/test_mse.cpp` checks the
known value, the gradient formula, and agreement with a numerical
gradient.

Because we sum over 784 pixels, the loss value looks large (tens,
early in training) compared to the classifier's cross-entropy (single
digits). This is a consequence of the summing-over-features
convention. What matters is that it decreases. Multiply by 2, divide
by 784, and take the square root to get a per-pixel RMS error in the
same [0, 1] units as the pixels.

---

## 2. Sigmoid activation

MNIST pixels lie in [0, 1]. If the decoder's last layer were linear,
it could output any real number, including negatives and values above
1, which are not valid pixel intensities. The logistic sigmoid maps
any real input into (0, 1):

```
sigma(x) = 1 / (1 + e^-x)
```

so the reconstruction is always a valid intensity. Its derivative has
a compact form:

```
sigma'(x) = sigma(x) (1 - sigma(x))
```

Since the forward pass already computed `Y = sigma(Z)`, the backward
pass reuses `Y` directly: `dZ = dY * Y * (1 - Y)`, element-wise. Like
ReLU, it is an element-wise activation, so the Jacobian is diagonal
and backward reduces to a Hadamard product. It has no parameters.

(Sigmoid appears again as the gate nonlinearity if we ever build
recurrent layers, and the same logistic shape is the binary case of
the softmax we already have.)

---

## 3. What an autoencoder is

An autoencoder consists of an encoder and a decoder:

```
encoder:  784 -> 128 -> ReLU -> 32 -> ReLU -> 16        (the bottleneck)
decoder:  16 -> 32 -> ReLU -> 128 -> ReLU -> 784 -> Sigmoid
```

The input (784 pixels) is reduced to a 16-number *latent code*, then
expanded back to 784 pixels. The loss is MSE between the output and
the original input. The training objective contains no labels; the
input is its own target.

The bottleneck is the essential part. 16 numbers cannot store 784
independent pixels, so the network is forced to find what is
redundant. MNIST digits are not random pixels. They lie on a
low-dimensional manifold (strokes, curves, a few degrees of freedom).
The autoencoder learns a coordinate system for that manifold. The
encoder maps an image to its coordinates, and the decoder maps
coordinates back to an image.

The bottleneck layer itself is **linear** (no activation). We want
the latent code to be able to take any real value in any direction.
Squashing it would distort the geometry that we use below for
interpolation.

---

## 4. Why two Networks instead of one stack

A plain autoencoder is a deep stack, so we could express it as a
single `Network` from 784 to 784. Building it as two separate
`Network`s, an encoder and a decoder, lets us call the halves
independently:

- `enc.forward(x)` → the latent code for an input
- `dec.forward(z)` → the image for a latent code

This separation makes the generative demos possible. A single stack
gives only input-to-output: there is no way to read the latent code
or to feed the decoder a code of our own choosing.

Training across two Networks needs no new library code. The latent
code is passed from one to the other:

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

The chain rule applies across the boundary: the decoder's backward
returns the gradient with respect to its input (the latent code),
which is the gradient the encoder's backward needs as its incoming
signal. One `Adam` instance optimises both networks. It keys its
per-parameter state on each weight matrix's address, so it can track
parameters from any number of networks at once.

---

## 5. Demo 1 — reconstruction

After training, we run a few test digits through `enc` then `dec` and
write the originals and their reconstructions to a BMP image
(`ae_reconstructions.bmp`) using the `Bmp.h` writer: top row the
originals, bottom row the reconstructions, each digit scaled up for
easier viewing. A well-trained autoencoder produces a slightly blurry
but clearly recognisable copy. The blur comes from the bottleneck:
fine detail is the high-dimensional information that 16 numbers can't
retain, so the network keeps the coarse shape and drops the rest. A
side effect is that the noise in any single pixel, being unpredictable
from the rest of the image, can't pass the bottleneck either, so
reconstructions come out denoised relative to their inputs.

---

## 6. Demo 2 — latent interpolation

This demo shows that the latent space is meaningful and not only a
means of compression. Take two test digits, say a 3 and an 8. Encode
both to get codes `za` and `zb`. Then walk a straight line between
them in latent space and decode each point:

```
z(t) = (1 - t) za + t zb,   for 8 evenly spaced t from 0 to 1
```

Decoding each `z(t)` produces a sequence of images that morph the 3
smoothly into the 8, written side by side into `ae_interpolation.bmp`.
The intermediate images are not averages of the two pictures, which
would look like a double exposure. They are plausible digit-like
shapes that the decoder generates from points the encoder never
visited during training. The network has learned a continuous space
in which the points between a 3 and an 8 decode to coherent shapes.

Generation relies on this continuity. If decoding arbitrary points in
latent space yields plausible outputs, then we can *sample* new points
and get new outputs: new digits, or in the audio version, new timbres.
The plain autoencoder comes close to this, with one limitation
described in the next section.

---

## 7. The limitation of the plain autoencoder

A plain autoencoder's latent space has no enforced shape. The encoder
is free to place codes wherever is convenient: clusters in some
places, voids in others, arbitrary scales on each axis. Interpolating
between two real encoded points usually works because the path tends
to stay near the data. But a random latent code with no real image
behind it often lands in one of the voids, and the decoder produces
meaningless output. The space is good for reconstructing and
interpolating known points, not for sampling new ones.

The **variational autoencoder** (next chapter) fixes this by forcing
the latent space into a known shape without gaps, a unit Gaussian, so
that any sample from that distribution decodes to something plausible.
That requires encoding a distribution instead of a point (the
reparameterisation trick), and a second loss term that pulls the
latent distribution toward the standard normal (the KL divergence).
Generation then consists of drawing `z` from a normal and decoding it.

Once generation works on MNIST, the same method applies to the audio
goal: train the encoder/decoder on per-frame spectra instead of
pixels, and the latent space becomes a space of timbres to interpolate
through and sample from.


## Further reading

Training a network to reproduce its input through a narrow layer, and
what the resulting codes are good for, is the subject of
[@hinton2006dim].
