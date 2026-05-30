# 22 — Positional encoding, LayerNorm, and the transformer block

[Note 21](21_attention.md) built attention as a pure operator on
`(Q, K, V)` and wrapped it in `MultiHeadAttention` with learnable
projections. By itself attention can't quite stand alone yet: it has
no notion of *where* a token sits in the sequence, no normalisation
to keep activations in a workable range across many layers, and no
internal feed-forward to give each position its own learned
transformation in between attention passes.

This note adds the three remaining pieces and assembles them into a
**transformer encoder block** — the basic repeating unit of a
transformer. None of the three ideas is deep, but skipping any one of
them in a real transformer training run produces something that
either won't learn or won't generalise. We'll build each one and end
with a single composable `Layer<T>` that wraps them all.


## 1. Why position matters

Attention is *permutation-equivariant* on the token axis. If you
shuffle the columns of the input, attention shuffles the columns of
the output the same way — but each individual output column is
unchanged. The output at position 3 doesn't know it's at position 3
versus position 7; it only knows what it asked for and what it found.

For images that's fine (a CNN treats translated patches the same), but
for sequences it's a problem. "The cat sat on the mat" and "Mat the
on sat cat the" should not produce the same set of token
representations. We need to inject the position into each token's
representation before attention sees it.


## 2. Sinusoidal positional encoding

The "Attention is All You Need" paper picked a beautifully simple
solution: add a deterministic, fixed pattern to each input token's
embedding, where the pattern depends only on the position.

For position `pos` and feature dimension `i`:

```
PE(pos, 2i)     = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1)   = cos( pos / 10000^(2i / d_model) )
```

Two pieces of intuition:

**Each position is unique.** The frequency progression
`10000^(2i/d_model)` covers a geometric sweep — at low `i` (early
features) the wavelength is short and the sine oscillates rapidly
with `pos`; at high `i` the wavelength is enormous and the sine is
nearly constant across the whole sequence. The combination of all
these frequencies is unique for every position, like a fingerprint.

**Relative offsets are linear.** Because `sin(a + b) = sin a cos b +
cos a sin b`, the encoding `PE(pos + k)` is a (position-independent)
linear function of `PE(pos)`. In principle the network can learn to
attend by *offset* — "look 3 positions back" — just from the
encoding, without needing to learn an explicit indexing scheme.

`SinusoidalPositionalEncoding<T>` precomputes the full
`(d_model × max_seq_len)` `PE` matrix at construction and just adds
it column-wise in forward. Backward is the identity: `d(X + PE)/dX
= I`, so `dX = dY`. No parameters, no training cost.

A learned alternative (a `(d_model × max_seq_len)` matrix of free
parameters, no sinusoids) is also common in modern transformers.
We'll add that variant if a downstream chapter needs it; for now the
sinusoidal version has the advantage of *extrapolating* to lengths
beyond the longest one seen in training (the formulas keep producing
distinct fingerprints).


## 3. LayerNorm

Stacking many attention blocks back to back has a tendency to drift
the activation magnitudes — each block adds and rescales things,
small errors accumulate. The original transformer paper used
**LayerNorm** to keep them in check; modern transformer training
without it is essentially never seen.

The formula is, per token (per column `j` of `X`):

```
mu_j      = (1/D) sum_i  X[i, j]
var_j     = (1/D) sum_i  (X[i, j] - mu_j)^2
x_hat[i,j]= (X[i, j] - mu_j) / sqrt(var_j + eps)
Y[i, j]   = gamma[i] * x_hat[i, j] + beta[i]
```

`D` is the feature dimension; `gamma` and `beta` are learnable
`D`-vectors initialised to 1 and 0 respectively (so the layer starts
as an exact standardisation and can drift to anything). The
`+ eps` (we use `1e-5`) keeps the divide safe when the variance is
near zero.

LayerNorm differs from BatchNorm in *which* axis the statistics come
from: BatchNorm uses the batch (column) axis, LayerNorm uses the
feature (row) axis. Transformers prefer LayerNorm because (i) it
works with any batch size including 1, (ii) it behaves identically at
train and inference time without needing running averages, and
(iii) the per-token statistics make sense for sequences of arbitrary
length where the "batch" might be a single sentence.

**The backward** is where LayerNorm earns its reputation for being
finicky. `X[k, j]` influences `x_hat[i, j]` through three paths:
directly (numerator), via the column mean `mu_j`, and via the column
variance `var_j`. After careful chain-ruling, with
`dx_hat[i, j] = dY[i, j] * gamma[i]`,
`S1_j = sum_i dx_hat[i, j]`, and
`S2_j = sum_i dx_hat[i, j] * x_hat[i, j]`, the input gradient is:

```
dX[i, j] = (1 / (D * sqrt(var_j + eps))) *
           ( D * dx_hat[i, j]  -  S1_j  -  x_hat[i, j] * S2_j )
```

`dgamma[i]` and `dbeta[i]` accumulate as `sum_j dY[i, j] * x_hat[i,
j]` and `sum_j dY[i, j]` respectively (the parameters are shared
across the batch, so their gradients sum across it). The
finite-difference test verifies all three to better than `1e-10` in
double precision, which is plenty to trust the implementation.


## 4. The transformer encoder block

With attention, positional encoding, and LayerNorm in place we can
assemble the standard encoder block. The composition has been the
subject of small but consequential variations; we use the **pre-LN**
variant, which is now the default in modern transformer
implementations because it trains more stably:

```
X --+--> LayerNorm --> MultiHeadAttention --+--> Y
    |                                       |
    +-----------------------(residual)------+

Y --+--> LayerNorm --> Dense -> ReLU -> Dense --+--> Z
    |                                            |
    +-----------------------(residual)-----------+
```

Two normalised sub-stacks (attention, feed-forward), each wrapped in
a residual connection. The feed-forward sub-stack is a standard
two-Dense bottleneck — `d_model → d_ff → d_model`, with `d_ff` usually
`4 * d_model`. The 4× expansion gives the per-position transformation
enough capacity to do interesting work; the ReLU is the only
nonlinearity in the sub-stack.

The **residual connections** are the load-bearing detail. Without
them, training a stack of more than a handful of these blocks
fails — gradients vanish through the long chain of matmuls and
softmaxes. With them, the gradient gets an identity shortcut that
keeps signal flowing, and the network can train hundreds of layers
deep. It's the same trick as ResNet, in the same role.

In code, each block is one `Layer<T>` that owns six sub-layers in
fixed order:

```cpp
LayerNorm<T>            ln1_;
MultiHeadAttention<T>   mha_;
LayerNorm<T>            ln2_;
Dense<T>                ff1_;
ReLU<T>                 relu_;
Dense<T>                ff2_;
```

Forward chains them with the two residual sums. Backward chains them
in reverse, with the residuals contributing identity gradients that
sum with the sub-stack gradients at the residual junctions:

```cpp
Matrix<T> backward(const Matrix<T>& dZ) {
    // FFN sub-stack:  Z = Y + FFN(LN2(Y))
    Matrix<T> df  = dZ;
    Matrix<T> de  = ff2_.backward(df);
    Matrix<T> dd  = relu_.backward(de);
    Matrix<T> dc  = ff1_.backward(dd);
    Matrix<T> dY_from_ffn = ln2_.backward(dc);
    Matrix<T> dY  = dZ + dY_from_ffn;

    // Attention sub-stack:  Y = X + Attn(LN1(X))
    Matrix<T> db  = dY;
    Matrix<T> da  = mha_.backward(db);
    Matrix<T> dX_from_attn = ln1_.backward(da);
    return dY + dX_from_attn;
}
```

The finite-difference check on `dX` agrees with the analytical
gradient to about `1e-8` in double precision — looser than the
individual sub-layer checks (which were near `1e-11`) because numerical
error accumulates across the six-layer chain, but well within the
range that catches real bugs.


## 5. What we have

Counting the moving parts of one encoder block at `d_model = 512`,
`n_heads = 8`, `d_ff = 2048`:

- Two LayerNorms: `2 * (512 + 512)` = 2,048 parameters
- One MultiHeadAttention: four `(512, 512)` projections with bias =
  `4 * (512 * 512 + 512)` = 1,050,624 parameters
- One feed-forward: `(2048, 512) + (512, 2048)` projections with bias
  = `2048*512 + 2048 + 512*2048 + 512` = 2,099,712 parameters
- **Total per block: ~3.15 million parameters**

For comparison, our ConvVAE for audio (note 20) had ~2 million total
across the whole encoder + decoder. A single encoder block at typical
transformer sizes already outweighs the entire ConvVAE. This is the
order-of-magnitude scaling that defines modern deep learning: not
because the *operations* are any deeper, but because the
*architectures* duplicate small blocks dozens of times. Twelve of
these blocks stacked is a small transformer (~38M parameters);
ninety-six of them stacked is GPT-2 large.


## 6. What's next

Two directions from here.

**The decoder side** of a full encoder–decoder transformer needs
*causal* masked self-attention (each position attends only to earlier
ones), and a *cross-attention* layer (the decoder queries the encoder
output). Both are minor extensions of what we have: the score matrix
gains an additive `-inf` mask before softmax, and the cross-attention
just takes `Q` from one source and `K, V` from another. We've
already separated those inputs in the SDPA interface, so it's a usage
pattern, not new code.

**The applied direction** is what closes Part II: a generative audio
system that combines matching pursuit (note 3) with attention. The
form is up for design — attention selecting MP atoms from a learned
dictionary, attention over MP residuals, attention as a re-ranking
step after greedy MP. Whatever the choice, it brings the two
generative families we've built (dictionary-based and
representation-based) into one place and points toward the chapters
of Part III, which will be about creative applications rather than
mechanism.
