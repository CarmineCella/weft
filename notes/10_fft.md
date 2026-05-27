# 10 — The FFT

We're about to do audio. Step one is being able to look at sound in the
frequency domain — what mix of sinusoids is in this signal? — and the
tool for that is the discrete Fourier transform. The FFT (fast Fourier
transform) is the same transform, computed cleverly enough to be useful
in practice.

## 1. The DFT

The discrete Fourier transform of an N-sample complex signal is

```
X[k]  =  sum_{n=0..N-1}  x[n] * exp(-2*pi*i * k * n / N)        k = 0, ..., N-1
```

Read it like this: bin `X[k]` measures how much of the signal looks like
a complex sinusoid at frequency `k/N` cycles per sample. The magnitude
`|X[k]|` says "how much of that frequency is here", and the phase
`arg(X[k])` says "where its peaks line up in time".

For a real audio signal sampled at 44.1 kHz with N = 4096, bin `k`
corresponds to frequency `k * 44100 / 4096 ≈ k * 10.77 Hz`. Bin 0 is
DC; bin 2048 is Nyquist (22.05 kHz); bins 1..2047 are positive
frequencies; bins 2049..4095 are the mirror image (negative frequencies)
which carry no extra information for real input. That's why we only
keep `N/2 + 1` bins downstream.

## 2. Why we don't compute the DFT directly

The formula above is an `O(N^2)` algorithm: N output bins, each a sum
over N inputs. For N = 4096 that's about 17 million complex multiply-
adds *per FFT*. With 5,000 audio files and tens of frames per file, the
cost balloons fast.

The FFT computes the same answer in `O(N log N)` operations. For
N = 4096, that's about 50,000 complex multiplies instead of 17 million —
a 350× speedup that's the entire reason audio DSP is practical.

## 3. The Cooley-Tukey trick

The key observation: a length-N DFT can be expressed as two length-N/2
DFTs glued together. Specifically, separate `x` into its even-indexed
and odd-indexed samples,

```
E[k] = DFT of x[0], x[2], x[4], ...     (length N/2)
O[k] = DFT of x[1], x[3], x[5], ...     (length N/2)
```

Then for k = 0 .. N/2 - 1:

```
X[k]         =  E[k]  +  w^k * O[k]
X[k + N/2]   =  E[k]  -  w^k * O[k]
```

where `w = exp(-2*pi*i / N)` is the "twiddle factor". Each step
combines two half-sized DFTs into a full one in `O(N)` extra work,
so the total cost satisfies

```
T(N)  =  2 * T(N/2)  +  O(N)         =>         T(N) = O(N log N)
```

If N is a power of two, we recurse all the way down to N = 1 (where the
DFT is the identity). That's the algorithm.

## 4. From recursive to iterative

The recursive version allocates new arrays at every level, which is
slow. The iterative version observes that all of those even/odd splits
can be done up front: if you reorder the input by reversing the bits of
each index, then the recursion's base cases sit in consecutive memory
slots, and you can build up the full transform by pairing adjacent
slots and growing the pair size.

That's what's in `FFT.h`:

```
1. Bit-reverse permutation:  O(N)
2. for len = 2, 4, 8, ..., N:
       for each block of `len` samples:
           combine two halves with twiddle factors
```

The outer loop runs `log2(N)` times; the inner loops together touch
each sample once per stage. Total: `N/2 * log2(N)` complex multiplies,
`N * log2(N)` complex adds, no allocations.

## 5. The inverse

The IDFT differs from the DFT by a sign in the exponent and a 1/N
factor:

```
x[n]  =  (1/N) * sum_{k=0..N-1}  X[k] * exp(+2*pi*i * k * n / N)
```

Rather than write a near-duplicate of `fft()` with the sign flipped,
we use a small identity:

```
ifft(x)  =  (1/N) * conj( fft( conj(x) ) )
```

Conjugating flips the sign of every imaginary component, which is
exactly the sign change in the exponent. So three lines and a reuse of
the forward FFT is all we need.

## 6. What we'll actually use it for

In the instrument classifier:

1. Cut each WAV file into overlapping frames (~4096 samples each).
2. Multiply each frame by a window function (Hann) to taper the edges
   to zero — without this, the frame boundaries look like step
   discontinuities and smear energy across the spectrum.
3. Lift the windowed frame from real to complex (zero imaginary part)
   and FFT it.
4. Take magnitudes of the first `N/2 + 1` bins.
5. Average across frames, weighted by per-frame energy, to get one
   "spectral envelope" per file.

The classifier then learns instrument identities from those envelopes.
Steps 1-5 are the subject of the next note; this one is just the FFT
itself, and the tests confirm it correctly returns:

- the DC magnitude `N` for a constant input,
- magnitude peaks at bins `k` and `N-k` for a cosine at frequency `k/N`,
- conjugate-symmetric output for real input,
- the input back when fed through `ifft(fft(.))`.

Anything that depends on FFT correctness — and lots of things will —
rests on those.
