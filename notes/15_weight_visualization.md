# 15 — Visualising weights

So far we've judged networks by their outputs — accuracy, reconstructions,
samples. But we can also look directly at what a network *learned* by
rendering its weights as images. For the first layer of an MNIST model this
is especially vivid, and it needs nothing we haven't already built: the
`Bmp.h` writer from note 13 does all the work.

## 1. Weights are images

A Dense layer's weight matrix `W` has one **row per output neuron** and one
**column per input**. For an MNIST model the input is 784 pixels, so each
row is 784 numbers — and 784 numbers reshaped to 28x28 is just an image.
Render that image and you see, literally, the pattern that neuron responds
to. The neuron computes a dot product of its weight row with the input, so
the row *is* the template it's matching against.

## 2. The linear classifier: ten templates

The example uses the simplest model that has interpretable weights: a linear
classifier, `Dense(784, 10)` followed by `Softmax`, with no hidden layer.
Its weight matrix is 10x784 — ten rows, one per digit class — so each row is
a per-digit **template**.

A linear classifier's decision is nothing but ten template matches: it dots
the input image against each of the ten weight rows (plus a bias) and picks
the largest. Softmax then turns those ten scores into probabilities. So the
templates aren't a metaphor — they are exactly what the model compares your
image to. Visualising them shows the entire "reasoning" of the model in ten
pictures.

On real MNIST the templates look like ghostly digit prototypes: the "0"
template is a ring of positive weight with a negative hollow in the middle
(ink in the centre is evidence *against* a zero); the "1" is a strong
vertical bar; and so on. Where a template is positive, ink at that pixel is
evidence *for* the class; where negative, ink there is evidence *against*.

(On trivially separable data — like the synthetic shapes used for smoke
testing — the templates come out noisy, because a model that hits 100%
accuracy in a few epochs never has to commit to a clean template. The
interpretable prototypes are a property of *real*, overlapping data forcing
the model to find genuinely discriminative pixels.)

## 3. Why a diverging colormap

Weights are signed — positive (evidence for) and negative (evidence
against) — and zero is meaningful (this pixel is irrelevant to this class).
A plain black-to-white ramp would hide the sign. So we use a **diverging
colormap**: white at zero, reddening toward positive, bluing toward
negative, normalised by the largest absolute weight so the scale is shared
across all ten templates. This is exactly why `Bmp.h` writes 24-bit RGB
rather than greyscale — the colour carries the sign, which is half the
information.

## 4. Reaching the weights

Two small things make this possible, both already in the library:

- `Network::add<L>(...)` returns a *reference* to the layer it just
  constructed. So we grab the Dense layer at build time —
  `Dense<float>& fc = net.add<Dense>(784, 10);` — and keep that reference.
  Because the network owns the layer by `unique_ptr`, the object's address
  is stable; the reference stays valid as more layers are added.
- `Dense::W()` exposes the weight matrix (const) for reading. After
  training, `fc.W()` is the 10x784 matrix we render.

Each row is reshaped to 28x28, mapped through the diverging colormap, scaled
up, and tiled into a 5x2 grid written to `mnist_weights.bmp`.

## 5. Deeper networks

For a network *with* hidden layers, the same trick applies to the **first**
layer: each of its rows is still a 784-vector, still an image, and they tend
to look like localised stroke and edge detectors — little oriented strokes,
loops, and blobs that later layers combine. The deeper layers are not
directly visualisable this way, because their inputs are activations, not
pixels; seeing what *they* respond to needs fancier techniques (e.g.
optimising an input to maximally excite a chosen neuron). First-layer
weights are the honest, zero-machinery version of interpretability, and a
linear model is the purest case of all — there *are* no hidden layers to
hide behind, so the weights are the entire model.

This is also a nice closing demonstration of why we built `Bmp.h` when we
did: the same writer renders autoencoder reconstructions, VAE manifolds, and
now the learned weights themselves.
