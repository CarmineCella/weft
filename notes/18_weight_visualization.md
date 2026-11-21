# 18 — Visualising weights

So far we have judged networks by their outputs: accuracy, reconstructions,
samples. We can also look directly at what a network learned by rendering
its weights as images. For the first layer of an MNIST model this works
particularly well, and it needs nothing new: the `Bmp.h` writer from
chapter 16 is sufficient.

## 1. Weights are images

A Dense layer's weight matrix `W` has one row per output neuron and one
column per input. For an MNIST model the input is 784 pixels, so each row
is 784 numbers, and 784 numbers reshaped to 28x28 form an image. That image
shows the pattern the neuron responds to. The neuron computes a dot product
of its weight row with the input, so the row is the template it matches
against.

## 2. The linear classifier: ten templates

The example uses the simplest model that has interpretable weights: a linear
classifier, `Dense(784, 10)` followed by `Softmax`, with no hidden layer.
Its weight matrix is 10x784 (ten rows, one per digit class), so each row is
a per-digit **template**.

A linear classifier's decision consists of ten template matches: it takes
the dot product of the input image with each of the ten weight rows (plus a
bias) and picks the largest. Softmax then turns those ten scores into
probabilities. The templates are therefore what the model compares the
image to, and visualising them shows the entire computation of the model in
ten pictures.

On real MNIST the templates look like faint digit prototypes: the "0"
template is a ring of positive weight with a negative region in the middle
(ink in the centre is evidence against a zero), the "1" is a strong
vertical bar, and so on. Where a template is positive, ink at that pixel is
evidence for the class; where it is negative, ink there is evidence against.

(On trivially separable data the templates come out noisy, because a model
that reaches 100% accuracy in a few epochs never has to settle on a clean
template. The interpretable prototypes appear because real, overlapping
data force the model to find discriminative pixels.)

## 3. Why a diverging colormap

Weights are signed, positive (evidence for) or negative (evidence against),
and zero is meaningful (the pixel is irrelevant to the class). A plain
black-to-white ramp would hide the sign. So we use a **diverging
colormap**: white at zero, increasingly red toward positive, increasingly
blue toward negative, normalised by the largest absolute weight so the
scale is shared across all ten templates. This is why `Bmp.h` writes 24-bit
RGB and not greyscale: the colour carries the sign, which is half the
information.

## 4. Reaching the weights

The library already provides what is needed:

- `Network::add<L>(...)` returns a reference to the layer it constructed.
  We take the Dense layer at build time,
  `Dense<float>& fc = net.add<Dense>(784, 10);`, and keep that reference.
  Because the network owns the layer by `unique_ptr`, the object's address
  is stable, and the reference stays valid as more layers are added.
- `Dense::W()` exposes the weight matrix (const) for reading. After
  training, `fc.W()` is the 10x784 matrix we render.

Each row is reshaped to 28x28, mapped through the diverging colormap, scaled
up, and tiled into a 5x2 grid written to `mnist_weights.bmp`.

## 5. Deeper networks

For a network with hidden layers, the same method applies to the first
layer: each of its rows is still a 784-vector and so an image, and the rows
tend to look like localised stroke and edge detectors (small oriented
strokes, loops, and blobs) that later layers combine. The deeper layers
cannot be visualised this way, because their inputs are activations, not
pixels. Seeing what they respond to needs more elaborate techniques (e.g.
optimising an input to maximally excite a chosen neuron). First-layer
weights are a form of interpretability that needs no extra machinery, and a
linear model is the simplest case: it has no hidden layers, so the weights
are the entire model.

The same `Bmp.h` writer renders the autoencoder reconstructions, the VAE
manifolds, and the learned weights.


## Further reading

Looking at what a network's weights and features respond to is developed
much further, for convolutional networks, in [@zeiler2014visualizing].
