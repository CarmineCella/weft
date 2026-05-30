# 21 — Attention

The library so far has two ways to mix information across positions in
an input. A *dense* layer mixes them all-to-all with a fixed weight
matrix that has to be learned exhaustively for every pair, which is
parameter-greedy and translation-blind. A *convolution* mixes only
neighbours within a fixed receptive field, which exploits locality
brilliantly when locality is the right prior, and not at all when it
isn't. Both share a hidden assumption: which positions interact, and
how much, is set at *layer construction time*, baked into the weight
shapes. Once you fix the network, the relationships are fixed.

Attention drops that assumption. Each position decides at *forward
time*, from the data, which other positions it should listen to. The
mechanism is small: a softmax over similarity scores. The
consequences are large enough to have replaced almost everything else
for language and a growing share of music and audio.

This note builds the attention operator from the ground up:
scaled-dot-product attention as a pure function of `(Q, K, V)`, then
multi-head attention as a `Layer<T>` that learns the projections.
Note 22 will add the surrounding pieces (positional encoding,
LayerNorm, a transformer encoder block).


## 1. Queries, keys, values

The unit of attention is a *retrieval*. Think of a dictionary lookup
where the comparison is fuzzy. You have a *query* vector and a set of
key/value pairs. The query is compared against every key; the closer
the match the more that key's *value* is included in the result. With
hard equality this is just a dictionary; with a soft, differentiable
comparison it becomes a generalisation of one.

Concretely, for one query `q ∈ R^{d_k}` and a set of `n_k` keys
`{k_j}` and values `{v_j}`:

```
score_j  =  <q, k_j>                          (a scalar similarity per key)
weight_j =  softmax(score_j over all keys)    (a probability per key)
out      =  sum_j  weight_j * v_j              (weighted combination)
```

That's it. Two operations: a similarity (a dot product), and a
weighted sum. The softmax in between is what makes it differentiable
and what guarantees that the weights sum to 1 — the result is a
*convex combination* of the values, so the output stays in the value
space.

For a *batch* of queries (in our case, every token in a sequence is
its own query), we stack them as columns of a matrix `Q ∈ R^{d_k ×
n_q}`. Keys and values stack as columns of `K ∈ R^{d_k × n_k}` and
`V ∈ R^{d_v × n_k}`. The whole batched operation collapses to two
matrix multiplications with a per-column softmax between them:

```
S = K^T Q                  (n_k × n_q)         every column = scores for one query
A = softmax(S, per column) (n_k × n_q)         every column = attention weights for one query
out = V A                  (d_v × n_q)         every column = weighted value combination
```

In our column-as-example convention this lands cleanly: each column
of the output is one query's retrieval.


## 2. The scaling

The formula above is the "raw" attention. Scaled-dot-product attention
divides the scores by `sqrt(d_k)` before the softmax:

```
A = softmax( (K^T Q) / sqrt(d_k) )
```

The reason is statistical. If `q` and `k` are independent vectors
with unit-variance components, then `<q, k>` has mean 0 and variance
`d_k`. For large `d_k` the dot product spreads over a wide range, and
the softmax becomes nearly one-hot: a tiny perturbation in the score
flips the weight by a lot, and the gradient through softmax collapses
to almost zero almost everywhere. The `sqrt(d_k)` rescaling keeps the
score variance at 1 regardless of dimension, so the softmax stays in
its "soft" regime where gradients flow. The Vaswani paper introduced
the scaling for exactly this reason, and it's the difference between
training that works and training that doesn't past `d_k ≈ 64`.


## 3. Backward

The forward pass has two matmuls and a softmax sandwiched between
them. The backward is the same in reverse: matmul backward (which we
already had in spirit since `Z = W X` plus its gradient is just
`dW = dZ X^T, dX = W^T dZ`) and softmax backward (which `Softmax<T>`
implements column-wise, exactly the form we need).

For `S = (K^T Q) / sqrt(d_k)`, `A = softmax(S)`, `out = V A`, given
`dOut := dL/dout`:

```
dV = dOut A^T                                   (d_v × n_k)
dA = V^T dOut                                   (n_k × n_q)
dS = softmax_backward(dA, A)   per column,
     dS[k, i] = A[k, i] * (dA[k, i] - sum_j A[j, i] dA[j, i])
dQ = (K  @ dS  ) / sqrt(d_k)                    (d_k × n_q)
dK = (Q  @ dS^T) / sqrt(d_k)                    (d_k × n_k)
```

We pack this in `ScaledDotProductAttention<T>` as a plain struct with
its own `forward(Q, K, V)` and `backward(dOut)` returning a tuple
`(dQ, dK, dV)`. It owns a `Softmax<T>` instance internally, which
caches `A` for its own backward — that's how we get the
softmax-backward step for free.

The numerical-gradient check passes at the ~1e-11 level in double
precision for all three gradients, which is the bar the library uses
to declare a backward correct.


## 4. Why three inputs, why not a Layer

You may have noticed that `ScaledDotProductAttention` is *not* a
`Layer<T>`. The library's layer abstraction promises a single matrix
input and a single matrix output. Attention has three inputs and one
output, and although you could stack `Q`, `K`, `V` vertically into one
big matrix and have the layer split it internally, doing so muddies
the gradient routing and forces a specific layout choice on every
caller. Instead we leave the bare operator as a non-Layer helper and
wrap it in a Layer one level up.


## 5. Multi-head attention

A single attention operation produces one weighted combination per
query. With one set of `(Q, K, V)` you get one "view" of the
relationships between positions. Multi-head attention runs `H` such
operations in parallel on disjoint slices of the feature axis, then
concatenates and re-mixes the result. Each head can attend to
something different — one might learn syntactic adjacency, another
long-distance reference, another a position-independent pattern — and
the model gets to use all of them.

The wiring, for input `X ∈ R^{d_model × seq_len}` (one token per column):

```
Q = Wq X                       (d_model × seq_len)
K = Wk X
V = Wv X
  split feature axis into H heads (each of size d_head = d_model / H)
  for h in 0..H-1:
    Q_h = rows [h*d_head, (h+1)*d_head) of Q     (d_head × seq_len)
    K_h, V_h similarly
    Y_h = ScaledDotProductAttention(Q_h, K_h, V_h)
  concat the Y_h back along the feature axis  (d_model × seq_len)
Y = Wo (concat)
```

The four projections `Wq, Wk, Wv, Wo` are the only learnable
parameters; everything else is the deterministic split-attend-concat.
We implement them as four full `Dense<T>` layers because the
column-as-example convention means each token in the sequence is
naturally one "example" — so `Dense(d_model, d_model)` does the right
thing without modification, and we inherit He initialisation, biases,
save/load and the optimizer hook from `Dense`. Each head gets its own
`ScaledDotProductAttention` instance, which is what isolates the
softmax caches so per-head backward works independently.

Gradient routing through MHA needs one care: in self-attention, `X`
is the *common* input to `Wq`, `Wk`, and `Wv`, so the gradient
returning from the three projection backwards must be summed:

```cpp
Matrix<T> dXq = Wq_.backward(dQ);
Matrix<T> dXk = Wk_.backward(dK);
Matrix<T> dXv = Wv_.backward(dV);
return dXq + dXk + dXv;
```

It's a one-line detail that's easy to miss, and a perfect example of
why finite-difference checks matter: a missing summand here would
pass every shape test, run without warnings, and quietly train a
broken model.


## 6. Two implementation traps we hit

These aren't deep ideas; they're the kind of thing the test suite
catches and that you'd otherwise spend an afternoon debugging.

**SIGFPE in the constructor.** Our first version of
`MultiHeadAttention(d_model, n_heads)` had `d_head_(d_model /
n_heads)` in the member-initializer list, with the validation `if
(n_heads == 0) throw ...` in the body. Member initializers run
*before* the body. Integer division by zero on Linux is `SIGFPE` (a
hardware trap), not a C++ exception, so passing `n_heads = 0` crashed
the test executable before any of our error messages could fire. The
fix is to do the validation in a static helper called from the
initializer list, so the throw lands first.

**One softmax cache per head, not one shared.** Each head needs to
remember *its own* attention weights for its backward pass, because
softmax backward formulas depend on the forward output. If multiple
heads shared a single `Softmax<T>` instance, the second head's
forward would overwrite the first's cache, and the first head's
backward would silently use the wrong weights. We give each head its
own `ScaledDotProductAttention` (which in turn owns its own
`Softmax<T>`); the per-head independence falls out of object
ownership.


## 7. What's not here yet

Attention as written above is *bidirectional* — every query attends
to every key in the sequence, no constraints. Three useful variants
need a small extension to support:

- **Causal masking.** For language models that predict the next
  token, position *i* must not attend to positions `> i`. The fix is
  to set `S[j, i] = -inf` for `j > i` *before* the softmax (so those
  weights become zero). One mask matrix, one entrywise add.
- **Padding masking.** Variable-length sequences in a batch get
  padded to a common length; the padded positions shouldn't
  contribute to anyone's attention. Same trick: `-inf` in the score
  matrix at padded positions.
- **Encoder–decoder cross-attention.** The query comes from the
  decoder's previous layer; the keys and values come from the
  encoder's output. Same operator, different sourcing of `(Q, K, V)`.
  We've already separated `Q` from `K, V` in the API, so this is just
  a usage pattern, not new code.

These all fit on top of `ScaledDotProductAttention` without changing
its interface. Note 22 builds the surrounding pieces — sinusoidal
positional encoding, LayerNorm, residual connections — and assembles
a complete transformer encoder block, with all of those features in
position to be added when we need them.
