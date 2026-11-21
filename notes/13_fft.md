# 13 — The FFT

The next chapters work with audio. The first step is to look at sound
in the frequency domain, that is, to find which mix of sinusoids a
signal contains. The tool for that is the discrete Fourier transform.
The FFT (fast Fourier transform) computes the same transform fast
enough to be useful in practice.

## 1. The DFT

The discrete Fourier transform of an N-sample complex signal is

```
X[k]  =  sum_{n=0..N-1}  x[n] * exp(-2*pi*i * k * n / N)        k = 0, ..., N-1
```

Bin `X[k]` measures how much of the signal looks like a complex
sinusoid at frequency `k/N` cycles per sample. The magnitude `|X[k]|`
says how much of that frequency is present, and the phase `arg(X[k])`
says where its peaks line up in time.

For a real audio signal sampled at 44.1 kHz with N = 4096, bin `k`
corresponds to frequency `k * 44100 / 4096 ≈ k * 10.77 Hz`. Bin 0 is
DC, bin 2048 is Nyquist (22.05 kHz), and bins 1..2047 are positive
frequencies. Bins 2049..4095 are the mirror image (negative
frequencies), which carry no extra information for real input. For
this reason we keep only `N/2 + 1` bins downstream.

## 2. Why we don't compute the DFT directly

The formula above is an `O(N^2)` algorithm: N output bins, each a sum
over N inputs. For N = 4096 that's about 17 million complex
multiply-adds per FFT. With 1,500 audio files and more than a hundred
frames per file, the total cost becomes large.

The FFT computes the same answer in `O(N log N)` operations. For
N = 4096, that's about 50,000 operations instead of 17 million, a
350× speedup. This speedup is what makes audio DSP practical.

## 3. The Cooley-Tukey trick

A length-N DFT can be expressed as a combination of two length-N/2
DFTs. Separate `x` into its even-indexed and odd-indexed samples,

```
E[k] = DFT of x[0], x[2], x[4], ...     (length N/2)
O[k] = DFT of x[1], x[3], x[5], ...     (length N/2)
```

Then for k = 0 .. N/2 - 1:

```
X[k]         =  E[k]  +  w^k * O[k]
X[k + N/2]   =  E[k]  -  w^k * O[k]
```

where `w = exp(-2*pi*i / N)` is called the twiddle factor. Each step
combines two half-sized DFTs into a full one in `O(N)` extra work,
so the total cost satisfies

```
T(N)  =  2 * T(N/2)  +  O(N)         =>         T(N) = O(N log N)
```

If N is a power of two, the recursion continues down to N = 1, where
the DFT is the identity.

## 4. From recursive to iterative

The recursive version allocates new arrays at every level, which is
slow. The iterative version does all of the even/odd splits up front.
If the input is reordered by reversing the bits of each index, the
base cases of the recursion sit in consecutive memory slots, and the
full transform is built by pairing adjacent slots and growing the pair
size.

This is the version in `FFT.h`:

```
1. Bit-reverse permutation:  O(N)
2. for len = 2, 4, 8, ..., N:
       for each block of `len` samples:
           combine two halves with twiddle factors
```

The outer loop runs `log2(N)` times, and the inner loops together
touch each sample once per stage. The total is `N/2 * log2(N)`
butterflies, each consisting of one data multiply, one twiddle update
and two adds, with no allocations.

## 5. The inverse

The IDFT differs from the DFT by a sign in the exponent and a 1/N
factor:

```
x[n]  =  (1/N) * sum_{k=0..N-1}  X[k] * exp(+2*pi*i * k * n / N)
```

Rather than write a near-duplicate of `fft()` with the sign flipped,
we use an identity:

```
ifft(x)  =  (1/N) * conj( fft( conj(x) ) )
```

Conjugating flips the sign of every imaginary component, which is the
sign change in the exponent. The inverse therefore takes three lines
and reuses the forward FFT.

## 6. What we use it for

In the instrument classifier:

1. Cut each WAV file into overlapping frames (~4096 samples each).
2. Multiply each frame by a window function (Hann) to taper the edges
   to zero. Without this, the frame boundaries look like step
   discontinuities and smear energy across the spectrum.
3. Lift the windowed frame from real to complex (zero imaginary part)
   and FFT it.
4. Take magnitudes of the first `N/2 + 1` bins, and their logarithm
   `log(1 + |X|)`.
5. Average across frames, weighted by per-frame energy, to get one
   spectral envelope per file.

The classifier then learns instrument identities from those envelopes.
Steps 1-5 are the subject of the next chapter. This chapter covers
only the FFT, and the tests confirm that it returns:

- the DC magnitude `N` for a constant input,
- magnitude peaks at bins `k` and `N-k` for a cosine at frequency `k/N`,
- conjugate-symmetric output for real input,
- the input back when fed through `ifft(fft(.))`.

Much of the later code depends on the FFT being correct, and therefore
on these tests.


## Further reading

The fast Fourier transform in the form used here is [@cooley1965fft].
For the discrete Fourier transform itself and its properties, see
[@oppenheim2009dsp].
