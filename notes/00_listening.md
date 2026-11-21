# Listen before building

Start with one cello note. A player holds A4, with changes in
pitch, bow pressure, and brightness. We can keep the recording, pass
it through a spectral model, or describe it with the controls of a
synthesiser. Each description gives us a different sound to work with.

The [listening player](audio/index.html) opens locally with the book's
companion files. The numbered WAV links below provide the same examples.
Play the recording first, then return to it between comparisons. Listen
to the attack, the sustained partials, the background texture, and the
release separately.

## A note through a bottleneck {.unnumbered}

Listen to **01: [the cello recording](audio/v020/cello_original.wav)**,
**02: [an STFT round trip](audio/v020/cello_stft.wav)**, and
**03: [the dense VAE reconstruction](audio/v020/cello_vae.wav)**.

The second clip provides a reference for the analysis and synthesis
steps. We divide the recording into overlapping windows, take Fourier
transforms, and transform them back while retaining both magnitudes
and phases. With enough overlap, the interior of the signal is
recovered to numerical precision. The boundaries need separate care.
This reference uses the same window and hop as the VAE application;
its independent implementation and boundary measurements are recorded
with the experiment.

The third clip takes a longer route. A neural network compresses each
frame's 2,049 log-magnitudes into 32 latent values. A decoder expands
those values into another magnitude spectrum. We combine that spectrum
with the original frame's phase and reconstruct a waveform.

Ask which parts of the cello survive the compression. Does the bow
attack remain distinct from the sustained tone? Do the upper partials
keep their definition? What happens near the release? Keeping the
input phase supplies information from the recording, but changed
magnitudes can still produce an inconsistent set of frames.

The model trained on 1,433 TinySOL recordings. This cello recording
was reserved for validation before training. Other cello notes,
including other recordings at the same pitch, remain in the training
set. The example therefore asks about another recording of a known
instrument.

We build the transform in [chapter 13](13_fft.md), the features in
[chapter 14](14_audio_features.md), and the VAE and its application in
chapters [17](17_vae.md) and [19](19_audio_vae.md).

\clearpage

## Fitting a sound and learning a control rule {.unnumbered}

Return to clip 01, then listen to **04: [directly fitted synthesiser
controls](audio/v020/cello_fit.wav)** and **05: [network-predicted
controls](audio/v020/cello_network.wav)**.

Both clips use a harmonic-plus-noise synthesiser. Oscillators follow
multiples of a supplied fundamental frequency; filtered noise supplies
a second component. The adjustable quantities are amplitudes and
noise-band gains over time.

For clip 04, gradient descent adjusts those controls for this recording
alone. The target is available throughout the fit. This gives us an
empirical reference for what the sound model and this optimization
procedure can reconstruct.

For clip 05, one network supplies the controls for many notes. It
receives pitch, a local loudness history, and a learned instrument
vector. Its weights were trained on the same 1,433 training recordings
as the VAE, with this cello file held out. At inference we still
measure pitch and loudness from the cello recording. The network's
task is to turn those inputs into synthesis controls.

Compare the two routes at the attack and during the changing sustain.
A close direct fit would support using this synthesiser for the
recording. A larger difference from the network would then motivate
questions about the learned control rule, its inputs, and training.
If both fits miss a feature, inspect what the synthesiser can express.
These comparisons help decide which part of the system to investigate.

The VAE keeps the recording's phase; the synthesiser creates its own
oscillator phases and noise realization. Their waveform differences
have different causes. A spectral residual gives us one numerical
comparison; listening tells us whether a particular difference matters
for the intended use. [Chapter 24](24_differentiable_synthesis.md)
develops both synthesis experiments.

\clearpage

## Giving the model a gesture {.unnumbered}

Listen to **06: [a flute phrase](audio/v020/gesture_phrase.wav)** and
**07: [a flute-to-cello transition](audio/v020/gesture_morph.wav)**.

The phrase follows eight prescribed notes with fixed timing and
loudness envelopes. The network supplies synthesis controls as those
inputs change. Listen especially at the joins: a collection of
sustained-note recordings leaves questions about attacks,
articulation, and movement between pitches.

The second gesture holds A4 while moving between two learned
instrument vectors. Listen for the points at which the change becomes
noticeable and whether the transition suggests a useful musical
material. The path and its timing are prescribed. The network supplies
a learned mapping along that path.

This gives the earlier comparisons a purpose beyond imitation. We
can ask how a representation helps us control a sound: which changes
are available directly, which require a model to infer something, and
which information has already been discarded. Later chapters ask
related questions using sequences of notes and lists of
time-frequency atoms.

The listening copies match whole-clip RMS within each reconstruction
group, with a common peak constraint. This reduces level differences
while leaving perceived loudness free to differ. The VAE application
already applies fades and normalization to its output; those steps
remain part of its result. The phrase and morph are separate gestures.
The [file guide](audio/README.md) records sources and processing.

All these clips were regenerated with weft v0.2.0. The fixed settings,
file splits, commands, logs, and measurements are in the
[experiment record](../book/experiments/README.md). MNIST provides
a separate numerical control in [chapter 12](12_mnist.md). Keep the
listening questions in mind as we build its matrices and gradients:
the same operations will eventually determine what we can hear.
