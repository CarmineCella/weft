# 21 — Differentiable synthesis: a network that plays a synthesiser

The audio VAEs of notes [16](16_audio_vae.md) and
[20](20_convvae.md) work, in the sense that they train, their latent
spaces are smooth, and the three applications run. But listen to the
results next to the recordings they came from and the verdict is
plain: they sound poor. The reason is not a bug and more epochs would
not fix it. We asked a small network to *regenerate a spectrum* from
32 numbers, and then we had to turn that spectrum back into a
waveform. Each step costs fidelity: squared error makes the network
output the average of plausible spectra, which smears sharp harmonic
peaks into a haze; the magnitudes are paired with the phases of a
different sound; and in the convolutional model 64 mel bands are
stretched back to 2049 bins.

This note takes a different route to the same goal. Instead of asking
the network for sound, we ask it for the *controls* of a synthesiser
that can only make plausible sounds in the first place. The
synthesiser is classical signal processing: a bank of sinusoids at
multiples of a fundamental, plus filtered noise. It has no trainable
parameters. The network never outputs a single audio sample. But
because every operation in the synthesiser is differentiable, the
error measured on the *audio* can flow back through it and tell the
network how to change the controls.

The idea is known as differentiable digital signal processing, or
DDSP, and the synthesiser inside it is much older: it is the
harmonic-plus-noise model of spectral modelling synthesis. That is the
point of this note. It is the place where the neural networks we have
been building meet the signal processing that audio people already
know, and each does the part it is good at.


## 1. The harmonic part

A pitched instrument tone is, to a good approximation, a sum of
partials at `f0, 2 f0, 3 f0, ...`. What distinguishes a flute from a
cello playing the same note is how loud each partial is and how that
changes in time. So the harmonic synthesiser takes

```
f0[t]      fundamental frequency in Hz, one per control frame
A[k, t]    amplitude of harmonic k+1 at frame t      (K x T)
```

and produces

```
y[n] = sum_k  a_k[n] * sin( (k+1) * phi[n] )
phi[n] = 2 pi * cumulative sum of f0[n] / sample_rate
```

Controls arrive once every 512 samples (11.6 ms at 44.1 kHz) and are
interpolated linearly up to the sample rate. A harmonic whose
frequency reaches the Nyquist limit is silenced, so the output never
aliases. All of this is `HarmonicSynth<T>`.

The backward pass is easy for a reason worth noticing: **the output is
linear in the amplitudes.** Write the interpolation weight of frame
`t` as `w_t[n]`; then

```
y[n] = sum_{k,t}  A[k,t] * w_t[n] * sin((k+1) phi[n])
```

and the gradient of any loss with respect to one amplitude is a
correlation of the upstream gradient with that amplitude's own basis
signal:

```
dA[k, t] = sum_n  dy[n] * w_t[n] * sin((k+1) phi[n])
```

No gradient flows to `f0`. Pitch is an input here, not something
learned (section 4 explains where it comes from).

One implementation detail: the sines are not computed with `sin()`.
From `sin(phi)` and `cos(phi)` the recurrence
`sin((k+1)x) = 2 cos(x) sin(kx) - sin((k-1)x)` gives every harmonic
with one multiplication each. And the running phase is kept in
`double` even when the library is instantiated with `float`: it is
multiplied by the harmonic number, and a float would lose exactly the
low bits that the high harmonics need.


## 2. The noise part

Not everything in an instrument tone is periodic. The breath of a
flute, the bow noise of a cello and the hiss inside an attack cannot
be built from harmonics. `FilteredNoise<T>` supplies them: white noise
shaped by a filter that changes over time.

```
M[b, t]    gain of frequency band b at frame t       (B x T)
```

The `B` bands (we use 65) are spread evenly from 0 Hz to Nyquist, and
the gain at an FFT bin is interpolated from the two nearest bands, so
`M` is a smooth spectral envelope with `B` control points. The filter
is applied in the short-time Fourier domain: cut a fixed, seeded noise
signal into overlapping windowed frames, FFT each, multiply by the
gains, inverse FFT, window again and overlap-add. Analysis and
synthesis both use a square-root Hann window, so that their product is
a Hann window and, at 50% overlap, unit gains return the noise
unchanged.

Like the harmonic part, the output is linear in the controls, and the
backward is the adjoint of the forward: window the upstream gradient
under each frame, FFT it, and correlate with that frame's noise
spectrum. Bins `k` and `N-k` share one gain, so their gradients add.

The two parts are simply summed. The whole synthesiser therefore has
`K + B` numbers per control frame, 165 in our configuration, and is
linear in all of them.


## 3. A loss the ear would agree with

Now we need to compare the synthesiser's output with a recording.
Comparing sample by sample is the wrong test. Two sine waves of the
same frequency and loudness but different starting phase sound
identical, yet their sample-wise difference can be as large as the
signals themselves. Our synthesiser starts every harmonic in sine
phase; a real instrument does not. A waveform loss would spend all its
effort on a difference nobody can hear.

What the ear compares is closer to the *magnitude* of the short-time
spectrum. `SpectralLoss<T>` does that. For one FFT size `N`, with a
Hann window and hop `N/4`:

```
S_y = | STFT(y) | / sum(window)          (same for the target x)
L_N = mean over all bins and frames of
        | S_y - S_x |  +  | log(S_y + eps) - log(S_x + eps) |
```

The linear term is dominated by the loud partials. The log term makes
quiet components count too: the decay of a note, the weak upper
harmonics, the noise floor.

The test suite shows the difference with three signals: a sine, the
same sine shifted in phase, and a sine at another pitch. The spectral
loss gives 0.198 for the phase shift and 2.83 for the pitch change.
The sample-wise error cannot tell the two cases apart.

One FFT size is a compromise: long windows resolve frequency and blur
time, short windows do the opposite. So the loss is summed over
several sizes. This is what "multi-scale" means.

**Backward.** For one frame the spectrum is

```
Y[k] = sum_i  y[i] * w[i] * exp(-2 pi j k i / N)
```

so

```
d|Y[k]| / dy[i]
    = w[i] * Re( conj(Y[k]) / |Y[k]| * exp(-2 pi j k i / N) )
```

Collect the per-bin loss gradients `g[k] = dL/d|Y[k]|` into
`C[k] = g[k] * conj(Y[k]) / |Y[k]|`, zero outside bins `0..N/2`. The
frame's gradient is then `w[i] * Re(FFT(C)[i])`, and the frames
overlap-add into the gradient of the whole signal. One FFT per frame,
the same cost as the forward. All three backward passes, and the chain
of synthesiser into loss, agree with finite differences to about
`1e-8` in `tests/test_synth.cpp`.


## 4. Pitch is an input

The synthesiser needs `f0`, and a small error there is expensive. If
`f0` is off by 10 cents (0.6%), partial 40 is off by the same 0.6%,
which at 17 kHz is 100 Hz: several FFT bins away from where the real
partial is. The synthesised partial and the recorded one then do not
overlap at all, and no choice of amplitude can make them match.

For the instrument samples we use, the note name is in the file name
(`Fl-ord-A4-mf.wav`), and `note_to_hz` turns it into a frequency. That
is correct to the nearest semitone only; players tune a little sharp
or flat and string players add vibrato. So `refine_f0` starts from the
nominal pitch and, frame by frame, searches 60 cents either side of it
for the fundamental whose harmonics best line up with the peaks of the
measured spectrum, by adding the spectrum's magnitude at
`f, 2f, 3f, ...` for each candidate. Because the search is confined to
less than a semitone around a known pitch there is no octave
ambiguity, which is the hard part of general pitch tracking and the
reason we do not need a full pitch tracker.


## 5. First experiment: no network at all

Before training a network to produce the controls, it is worth
asking how good the synthesiser can sound at all. `synth_fit` answers
that. It treats the controls themselves as the parameters: the
amplitude of every harmonic and the gain of every noise band at every
frame, about 60,000 numbers for a six-second note. The loop is the
one every example in this book uses, with the synthesiser where the
network would be:

```
controls -> HarmonicSynth + FilteredNoise -> audio
audio, recording -> SpectralLoss -> dL/d(audio)
dL/d(audio) -> synthesiser backward -> dL/d(controls) -> Adam
```

Whatever a network later predicts, it cannot do better than controls
fitted directly to the target. This is the ceiling.

```
synth_fit fit  Fl-ord-A4-mf.wav  A4  flute
```

Four hundred iterations take about two minutes per note. Measured on
the spectrogram, the difference between the resynthesis and the
recording is under 0.5% of the energy for flute, clarinet, trumpet,
violin and bassoon, and 3% for a cello note with wide vibrato. The
noise part turns out to be small for these sustained tones, under
half a percent of the energy.

It is important to say what this is. **It is analysis and
resynthesis**, the same family as the phase vocoder and sinusoidal
modelling: measure a sound's partials, then rebuild it from them.
Gradient descent is an unusual way to do the analysis, but nothing
here is beyond what classical methods already do, and
`synth_fit morph` and `synth_fit blend`, which interpolate between
the fitted controls of two notes, are the classical cross-synthesis
of two sets of partials. What the network adds comes in section 7.


## 6. Two traps, both about the period

The first fits of low notes were bad. A bassoon A2 came out with its
harmonics switching on and off from frame to frame, and a spectrum
that no longer matched the recording. The diagnosis is instructive because the first
three explanations were wrong.

It was not pitch: a long FFT shows the bassoon's partials at exactly
110.00 Hz. It was not the optimiser's settings. And it was not the
missing phases, which we could test directly with a synthetic note
that the synthesiser can reproduce *exactly* (twenty harmonics of
110 Hz, in sine phase). The fit of that note flickered too. So the
loss surface itself had bad solutions, and two things created them.

**Controls that move too fast.** A control frame lasts 11.6 ms; one
period of A2 lasts 9 ms. If a harmonic's amplitude may change as fast
as the waveform repeats, switching it on and off becomes another way
of shaping the waveform, and the optimiser uses it. The fix is to
store the controls at *knots* spaced so that one knot spans about four
periods of the note, and interpolate between them. For A4 the rule
changes nothing; for A2 it places a knot every fourth frame. With that
alone, the synthetic note's amplitudes came out steady.

**Windows that are too short.** A window shorter than a few periods
cannot separate neighbouring harmonics. What it measures then depends
on their relative *phases*, and those the synthesiser does not
control. Given a synthetic note with random phases, the fit returned
0.078 and 0.047 for two harmonics whose true amplitudes were 0.05 and
0.15: it had traded correct amplitudes for a better waveform shape.
The fix is to use only FFT sizes of at least 2.5 periods: 256 to 4096
for A4, 1024 to 8192 for A2. With both fixes the same note is
recovered as 0.050, 0.150, 0.300, 0.100, which are the true values,
and the bassoon's resynthesis matches the recording to 0.4%.

Both rules say the same thing: *the time scale of the model has to be
set by the pitch period, not by a fixed number of milliseconds.* They
carry over unchanged to the network.


## 7. The network

`synth_train` trains one network to produce the controls for any note
of any instrument. For every control frame it reads 26 numbers:

- **Pitch**: `log2(f0 / 440)`, scaled.
- **Loudness**: the level of the sound around the frame, in dB, at
  nine time offsets from 8 frames before to 8 frames after. One value
  would not be enough, because an attack and a decay pass through the
  same level and need different spectra.
- **Instrument**: a learned 16-dimensional vector, one per instrument.

The instrument vector is our first use of `Embedding<T>`, a lookup
table of trainable vectors indexed by an integer. Its forward copies
the selected column; its backward adds the incoming gradient into that
column. Nothing tells the network what a flute is: twelve vectors
start as random numbers and are moved by the same gradient descent as
every other parameter until they are useful.

The network itself is the plainest in the book, three hidden layers
of 256 units with ReLU, 181,000 parameters in all. It outputs 166
numbers per frame, which a small "head" turns into controls:

```
amp   = act(o[0])                         one overall amplitude
h_k   = act(o[1+k])   below Nyquist, else 0
A_k   = amp * h_k / sum_j h_j             harmonic amplitudes
M_b   = act(o[1+K+b])                     noise band gains
act(x) = 2 * sigmoid(x)^ln(10) + 1e-7
```

Splitting the harmonics into "how loud overall" and "how the loudness
is shared" lets the network learn timbre separately from dynamics.
The activation is always positive and close to exponential for small
values, so equal steps in the network's output are roughly equal
steps in decibels.

One training step takes a batch of notes, cuts a random 1.5-second
crop from each, runs the network once over all their frames, and then
runs the synthesiser and the loss for each note on its own thread. The
gradients come back through the synthesiser, through the head, into
the network and into the instrument vectors.

```
synth_train train  data/TinySOL_2020  synth_model  60
```

On TinySOL (1,508 usable notes of 12 instruments, 75 of them held
out) sixty epochs take 17 minutes. The loss falls from 3.25 to 2.19
on the training notes and from 2.87 to 2.37 on the held-out ones.


## 8. What the network adds

A direct fit can only rebuild the note it was fitted to. The network
has learned a *function* from pitch, loudness and instrument to
sound, and a function can be evaluated where there is no recording.

```
synth_train resynth synth_model  cello.wav  A4  Fl   out.wav
synth_train note    synth_model  Fl Vc  A4  6  flat  out.wav
synth_train note    synth_model  Vn Vn  A4  5  vibrato  out.wav
synth_train phrase  synth_model  ClBb  out.wav
```

- **Timbre transfer.** `resynth` takes the pitch and loudness curves
  of a recording and plays them with any instrument's vector: a cello
  note, with the cello's own vibrato and bow changes, as a flute.
- **Morphing.** Because an instrument is a point in a
  16-dimensional space, we can move between two points while a note
  is held. The intermediate timbres belong to no instrument.
- **Notes that were never played.** A crescendo, a vibrato, a
  glissando over an octave, a phrase of eight notes: the training
  data contains only single sustained notes, so all of these are
  new. So are notes outside an instrument's range, such as a flute at
  A2.

None of this is concatenated from recordings. Every sample comes out
of the oscillators and the noise filter.


## 9. What this is and is not

Compared with the VAEs the difference is immediate: the output is
clean. There is no phase problem because the synthesiser generates
coherent sinusoids, and no inversion problem because there is no
spectrogram to invert. The loudness envelope and the pitch, vibrato
included, are reproduced closely.

The timbre is less faithful than a direct fit, and the numbers say by
how much: 2.4 for the network on unseen notes against 1.0 to 1.6 for
the direct fits. The network has learned a somewhat generic version of
each instrument. On held-out notes the trumpet and trombone are close
to the recordings; the clarinet comes out with a full set of
harmonics although a real clarinet's even harmonics are weak; a
pianissimo flute is too bright; a fortissimo double bass is 7 dB too
quiet. A longer run would narrow the gap. It would not close it,
because a network that must explain every note with one function is
solving a harder problem than a fit of one note.

The limits of the approach are built into the synthesiser:

- **One pitched voice.** A chord, a drum, a noisy texture or a
  recording of several instruments is outside what a harmonic series
  plus filtered noise can describe.
- **Pitch has to be known.** We read it from the file name and refine
  it. Applying the model to an arbitrary recording needs a real pitch
  tracker.
- **It tends towards the well-behaved.** A harmonic synthesiser
  driven by smooth controls sounds like a very good additive
  synthesiser: stable, clean, and without the roughness of a real
  attack.

What we have is not a new kind of sound. It is a connection: the
synthesiser is the one signal processing has used for decades, the
training loop is the one we have used since note 06, and the only new idea is that
the first can be placed inside the second. The network supplies what
the classical method lacks, a mapping from musical intentions to
synthesis parameters, and the synthesiser supplies what the network
lacks, a guarantee that whatever comes out is a sound.

The next notes return to architecture. Everything so far has
processed one frame, one patch or one image at a time. Attention is
how a network relates the elements of a *sequence* to one another,
and it is what the generative models at the end of the book are built
from.
