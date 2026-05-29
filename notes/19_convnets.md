# 19 — Convolutional networks

For the first eighteen notes we worked exclusively with dense (fully
connected) layers. Dense layers handle the MNIST classifier, the Iris
classifier, the audio VAE on spectrograms — anything where the input is
naturally a flat vector of features and you don't care which feature
sits next to which. They are limited in a specific, important way: they
have no notion of *spatial structure*. A dense layer treats every input
neuron as exchangeable with every other one. Permute the columns of a
batch of MNIST images and a dense MLP, retrained on the permuted data,
will reach exactly the same accuracy. That is the giveaway that dense
layers are not using anything about the *2-D arrangement* of pixels.
For images, that's leaving most of the structure on the table.

This chapter is about the layer that does use it: **convolution**. The
goal is to motivate it honestly, derive its forward and backward math,
explain the engineering trick (`im2col` + matmul) that makes it run
fast on a CPU, and then train a small ConvNet on CIFAR-10 to see the
payoff — not just the accuracy number, but the *learned first-layer
kernels*, which are interpretable as images in their own right and tell
you what the network has actually decided "an edge" looks like.


## 1. Why dense MLPs are wrong for images

Three reasons, each separately damning:

**Parameter explosion.** A 32×32 RGB CIFAR-10 image has 32·32·3 = 3072
input pixels. A single dense layer with 1024 hidden units would have
3072 × 1024 ≈ 3.1 million weights in the *first layer alone*. Most of
those weights are answering questions like "what is the joint relevance
of pixel (3, 7, red) and pixel (28, 14, blue)?" — questions that for
natural images simply don't have meaningful answers. The weights are
mostly wasted.

**No translation invariance.** Suppose we have a perfectly trained
dense MLP that classifies cats correctly. We take a cat image, shift
every pixel one column to the right (and wrap the rightmost column to
the left, say). To a human, it's the same cat. To the dense MLP, every
single input has changed: the pixel that *was* at index 5 of the input
vector is now at index 6, and the weight matrix has no idea those two
positions are spatially adjacent. The MLP has to learn the concept of
"cat" *separately* at every possible translation. That's vastly
inefficient and, in practice, doesn't generalise.

**No exploitation of locality.** Natural images have a strong
property: pixels close together in space are far more likely to be
related than pixels far apart. Edges, textures, object parts — all
local. A dense layer cannot represent this prior; it has to discover
locality from data, using millions of weights that mostly should be
zero. The right thing is to *build the locality into the architecture*.


## 2. The convolution operator

A 2-D convolution at one output position computes

    y[h, w]  =  sum over (kh, kw) of  x[h + kh, w + kw] · k[kh, kw]

where `k` is a small kernel (3×3 or 5×5 are typical) and the same
kernel `k` is applied at every output position `(h, w)`. Three
properties fall out for free:

- **Local connectivity.** Each output value `y[h, w]` depends on a
  small neighbourhood of the input — only the K² pixels in the kernel's
  footprint — not on the whole image. A 3×3 kernel uses 9 input pixels
  per output, regardless of how big the image is. The parameter
  explosion goes away.

- **Weight sharing.** The same kernel `k` slides across the whole
  image. There are K² weights total *for the entire layer*, not per
  position. A 3×3 kernel has 9 numbers in it, period. (More precisely:
  K² × C_in × C_out, but the *spatial* part is K² regardless of image
  size.)

- **Translation equivariance.** Shift the input one pixel right; the
  output shifts one pixel right. The kernel didn't change, the
  computation didn't change. The network has the right structural
  prior for natural images baked in.

The third property is the deepest. Equivariance is not invariance —
the network's *output* shifts when the input shifts, it doesn't stay
the same. But that turns out to be what you want for the early layers,
because the *final* layers (after pooling and flattening) can become
invariant on top of equivariant features. Build up locality first,
then collapse it.

**Multi-channel extension.** Real images have channels (R, G, B), and
intermediate feature maps have many channels (32, 64, sometimes
hundreds). So the kernel is actually a 3-D object — `(C_in, K, K)` —
that's applied to a `C_in`-channel patch and produces *one* scalar
output. To produce `C_out` output channels, we use `C_out` independent
3-D kernels. So a conv layer with C_in=3, C_out=32, K=5 has

    32 × 3 × 5 × 5 = 2400 weights

plus 32 bias values (one per output channel). Tiny compared to a dense
layer's millions, and that's the point.


## 3. Padding and stride

Two knobs let convolution work with images of any size.

**Padding** adds zero-pixels around the input border so the output can
have the same spatial dimensions as the input. A 3×3 kernel with no
padding (`p = 0`) takes a 28×28 input to a 26×26 output (it can't be
centred on the boundary pixels). With `p = 1`, you get a 28×28 input to
28×28 output — "same" padding. Whether you want that depends on the
architecture; it's common to use it in early layers so the spatial
dimensions only shrink at the pooling steps, not inside the conv
itself.

**Stride** controls how far the kernel moves between output positions.
Stride 1 (the default) puts an output at every input pixel. Stride 2
puts one at every other pixel, which halves the output spatial size.

The combined output formula is

    H_out = (H_in + 2p - K) / s  +  1

and the same for W. A 32×32 input with K=5, p=2, s=1 gives
H_out = (32 + 4 − 5) / 1 + 1 = 32 — preserved. K=3, p=1, s=1 also
preserves. K=3, p=1, s=2 halves: H_out = (32 + 2 − 3) / 2 + 1 = 16.


## 4. im2col: making convolution fast on a CPU

The math above describes the *operation*. Writing it as a literal
implementation — six nested loops over (n, out_C, h_out, w_out, in_C,
kh, kw) — works, but is slow. The CPU's cache doesn't enjoy seven-deep
loop nests, and we've already invested significant work into making
`Matrix::operator*` fast (parallel rows, optional BLAS dispatch). It
would be nice to *reuse* the matmul for convolution.

We can. The trick is called **im2col**, short for image-to-column. The
idea: rearrange the input so that the convolution becomes a single
matrix multiplication.

Specifically: for every output position `(n, h_out, w_out)`, the input
patch under the kernel is a 3-D chunk of shape `(C_in, K, K)`. Flatten
that chunk into a column of length `C_in · K · K`. Stack the columns
side by side across all output positions:

    im2col matrix:  (C_in · K · K)  ×  (N · H_out · W_out)

Now reshape the kernel from `(C_out, C_in, K, K)` into a matrix of
shape `(C_out, C_in · K · K)` — one row per output channel, the row
being that channel's flattened kernel. Then

    Z = W · im2col            shape: (C_out, N · H_out · W_out)

and the entire convolution is *one* matrix multiply. We get all our
prior matmul optimisations (`std::thread` parallelisation, optional
BLAS) for free. This is exactly what cuDNN does on GPUs.

The cost is the memory of the temporary `im2col` matrix, which is K²
times larger than the input. For our scales — CIFAR-10 with small
batches — this is comfortably within memory. For very large images or
deep networks, more sophisticated schemes (Winograd, FFT-convolution)
become competitive; we don't need them.

**Backward via col2im.** The backward pass reverses the trick. Given
the gradient `dY` of shape `(N, C_out, H_out, W_out)`:

1. Reshape `dY` into `dZ` of shape `(C_out, N · H_out · W_out)`.
2. Parameter gradients are straightforward matmuls:
   `dW = dZ · im2col^T`  (kernel gradient),
   `db = sum-along-columns of dZ`  (bias gradient).
3. For the gradient w.r.t. the input, compute `dCol = W^T · dZ`. This
   has the shape of the original im2col matrix.
4. **col2im** folds `dCol` back into the input shape, *accumulating*
   the contributions at every input pixel that was touched by more
   than one output position. When stride is less than kernel size,
   that's most pixels.

The accumulation is the subtle part. If you write `=` instead of `+=`
in the col2im loop, the gradient computation is silently wrong for
every conv layer with stride < K. The numerical gradient check in
`tests/test_conv2d.cpp` is exactly what catches this kind of bug.


## 5. Pooling

After every conv layer (or every couple) we want to *reduce* the
spatial size — both because we want to build invariance to small shifts
on top of the equivariant features, and because it cuts the
computational cost of subsequent layers. The standard tool is
**max-pooling**: take a window (typically 2×2) and emit the maximum
value.

Forward is obvious. Backward is the part worth understanding: the
output depended on *one* input pixel per window (the one with the max
value). Every other pixel in the window contributed nothing, so the
gradient `dY` for that output cell flows entirely to the winning input
pixel and nowhere else. We store the winning index during forward and
look it up during backward.

With non-overlapping windows (stride = pool size, the typical
"halve everything" case), each input pixel wins at most once. With
overlapping pooling (stride < pool size), an input pixel can win in
multiple output windows, and the gradients accumulate. Same `+=` issue
as conv backward; the same test pattern catches it.

Max-pool has zero learnable parameters and runs in O(N · C · H · W).
Cheap, robust, and the workhorse of every conv classifier.


## 6. The conv-to-dense bridge

Putting convolution and pooling together: a small conv stack reduces a
32×32×3 image to a `(C, H', W')` feature map — say `(64, 8, 8)`. To
predict a class probability vector, we need a *vector* output, not a
3-D feature map. The standard move is **flatten**: turn the 3-D feature
map into a 1-D vector and feed it to a dense classifier.

In our library, this is the moment where the data structure changes —
`Tensor4D` becomes `Matrix`. We do this with an explicit free function
`flatten(Tensor4D) -> Matrix`, *not* a hidden reshape inside a unified
tensor type. The conv stack is one `ConvNetwork`; the dense head is
another `Network`; the example wires them together. Why this is the
right choice for a teaching library is discussed at length in
[note 18](18_from_matrices_to_tensors.md).

The classic AlexNet shape, in our library, looks exactly like this:

```cpp
ConvNetwork<float> conv;
conv.add<Conv2D>(  3, 32, 5, 1, 2);  conv.add<ReLU4D>();  conv.add<MaxPool2D>(2);
conv.add<Conv2D>( 32, 64, 3, 1, 1);  conv.add<ReLU4D>();  conv.add<MaxPool2D>(2);

Network<float> dense;
dense.add<Dense>(64 * 8 * 8, 10);
dense.add<Softmax>();
```

The forward pass at every training step:

```cpp
Tensor4D<float> feat  = conv.forward(X);            // (N, 64, 8, 8)
Matrix<float>   flat  = flatten(feat);              // (4096, N)
Matrix<float>   probs = dense.forward(flat);        // (10, N)
```

And the backward mirrors it exactly, with `unflatten` doing the reverse
reshape on the gradient.


## 7. What the library provides

The conv work in this chapter added five new headers:

- **`Tensor4D.h`** — the 4-D dense tensor in NCHW layout, with
  element access, `apply`, `fill`, randomisation, and a `hadamard`
  free function for element-wise products in backward computations.
  Covered in [note 18](18_from_matrices_to_tensors.md).

- **`ConvLayer.h`** — the abstract base class for conv-stack layers,
  parallel to `Layer` for the dense world.

- **`ConvNetwork.h`** — the container, parallel to `Network`.
  Supports `add<L>(args...)`, `forward`/`backward`/`update`,
  `summary()`, `save`/`load` (with a distinct `WCNV` magic so a conv
  file cannot be loaded into a dense network by mistake).

- **`Conv2D.h`** — the convolution itself, with He initialisation, the
  im2col forward and col2im backward, configurable kernel size, stride,
  and padding. Verified by numerical gradient check on `dX`, `dW`,
  `db` in two stride configurations.

- **`ReLU4D.h`**, **`MaxPool2D.h`**, **`Flatten.h`** — the supporting
  cast. `MaxPool2D` carries a numerical gradient check too, because
  its backward (routing through the cached argmax) is exactly the kind
  of indexing logic that's easy to get wrong.

The whole conv library lives behind tests that pass at full numerical
precision (double-precision gradient checks, 1e-6 relative tolerance).
If those pass, the math is right; if they fail, no architectural
experiment will rescue it.


## 8. The CIFAR-10 classifier

CIFAR-10 is the standard "step up from MNIST" image dataset: 60,000
32×32 colour images split into 50,000 training and 10,000 test, with
10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse,
ship, truck). It's exactly the right size to be challenging without
needing a GPU: a small ConvNet trained on CPU can reach ~75-80% test
accuracy in something like an hour.

We deliberately *skipped* MNIST for the conv classifier. The dense MLP
chapter already classified MNIST at ~98% test accuracy; a conv version
would just be making the same point in a more complicated way. Going
straight to CIFAR-10 shows the genuine win of convolution — on data
where a dense MLP struggles to get above ~50% accuracy, the small
ConvNet reaches well above 70%.

**The architecture** we use is `cifar_conv.cpp`:

```
input  (N, 3, 32, 32)
  Conv2D(3 -> 32, k=5, p=2)     -> (N, 32, 32, 32)     <-- 5x5 first layer
  ReLU4D
  MaxPool2D(2)                  -> (N, 32, 16, 16)
  Conv2D(32 -> 64, k=3, p=1)    -> (N, 64, 16, 16)
  ReLU4D
  MaxPool2D(2)                  -> (N, 64, 8, 8)
---- flatten ----                -> (4096, N)
  Dense(4096, 10)
  Softmax
```

Two design choices worth pointing at:

- **5×5 kernels in the first layer**, 3×3 in the second. The first
  layer is the one we visualise, and 5×5 RGB tiles are much more
  legible than 3×3 ones. The deeper layer can use the smaller kernel
  because it operates on richer features (32 channels rather than 3).
  This is the same logic that led Krizhevsky's AlexNet to use 11×11 at
  the first layer (working on 224×224 images), 5×5 next, then 3×3.

- **Two conv blocks**, not three. A third block would push accuracy
  slightly higher (78-82% range) but adds another hour of CPU training
  for a marginal improvement. The architecture above is the smallest
  one that demonstrates the full conv arc — locality, hierarchy,
  pooling, dense head — without the iteration time becoming
  frustrating. It's a pedagogical choice; a production CIFAR-10 model
  would use ResNet or something larger.

**Training.** Adam optimizer (lr = 1e-3, defaults), batch size 64, 15
epochs, He-initialised conv kernels, zero-initialised biases, no data
augmentation. Run it with

```sh
./bin/examples/cifar_conv  data/cifar-10-batches-bin  15  cifar_model
```

and on an Apple M3 with `make BLAS=1` it should take roughly 60-90
minutes and produce a model that reaches around 72-78% test accuracy.
The training prints loss, train accuracy, and test accuracy each epoch,
and saves a checkpoint after each one — Ctrl-C is safe.


## 9. The payoff: visualising what the first layer learned

The whole point of going to CIFAR rather than MNIST was the
visualisation. The companion example `cifar_filters.cpp` loads the
trained model, extracts the first conv layer's 32 5×5×3 filters, and
arranges them in a 4×8 grid BMP. Each filter is normalised
independently to fill [0, 255] so weak and strong filters are both
visible.

What you should see, after a successful training run:

- **Oriented edge detectors** at various angles (horizontal,
  diagonal, vertical). These look like dark/light gradients across the
  kernel — high values on one side, low on the other. The kernel is
  effectively a small derivative filter in some direction.
- **Color-opponent blobs.** Red-on-green, blue-on-yellow. These are
  the network discovering colour structure on its own — nothing in
  the training told it that RGB pixels have any particular meaning,
  but with enough natural images the colour-opponency falls out of
  what's discriminative.
- **Low-frequency centre-surround patterns.** Some filters look like
  blurry blobs, picking up overall brightness or colour in a region.
- **Maybe a few dead filters** — uniform grey tiles that never
  activated and never learned. This is normal; on a small budget of
  filters, a few often end up at random init.

This is exactly the picture Krizhevsky's AlexNet paper (2012) and
Zeiler & Fergus's *Visualizing and Understanding Convolutional
Networks* (2014) made famous. Our version, trained on CPU on a much
smaller architecture and dataset, shows the same qualitative pattern.
That's not a coincidence — the structure falls out of the operator and
the data, not the depth or scale.

The deeper layer is harder to visualise as raw weights, because its
filters operate on the 32-channel output of the first layer rather
than on RGB. Each second-layer kernel is `(32, 3, 3)` — not directly
an image. Techniques for visualising deeper layers (activation
maximisation, deconvolutional inversion) exist but are out of scope
here. For now, the point is made: the first layer learns *features*,
and those features look like edges and colour patches because that's
what is locally informative in natural images.


## 10. What's next

The conv arc continues into audio. The same Conv2D layer, applied to
log-magnitude spectrograms instead of CIFAR images, gives us a
**ConvVAE for sound** — the same generative architecture as the dense
audio VAE from [note 16](16_audio_vae.md), but with the magnitude-blur
problem of the dense version largely fixed by local spatial structure
in the encoder and decoder. That's [note 20](20_convvae.md) (still to
come): the architecture, the training, the morphs and interpolations,
and an honest comparison against the dense version. We should hear the
difference.

After ConvVAE the architecture path goes one more direction —
**attention** — and then the capstone, the matching-pursuit + attention
generative audio system. The work to get to that point is now mostly
mechanical, because the foundation is in place.
