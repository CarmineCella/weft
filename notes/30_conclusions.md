# 30 — Conclusions

The cello note at the beginning of this book now has several
descriptions. It is a waveform, a sequence of spectra, a series of
latent codes, and a set of synthesiser controls. Each
description makes some changes accessible and leaves other work
to the model or the renderer. Following those choices through to
sound has been the purpose of building weft.

## 1. What the implementation makes visible

A matrix product is involved in both classifying a handwritten digit
and predicting the amplitudes of a harmonic synthesiser. The difference
lies in the inputs, the objective, and what happens to the output.
Implementing the product, the layers, and their backward passes gives
us access to each of those boundaries. We can inspect which values
were cached, where a gradient is averaged, and how an error reaches
a parameter.

Numerical gradient checks give this inspection a practical test.
They compare a backward pass with small perturbations of the forward
calculation. Their evidence concerns the inputs and paths exercised
by the check. Data leakage, unsuitable loss functions, and a poor
audio representation require different experiments.

The three MNIST runs in [chapter 12](12_mnist.md) provide one
control: the corrected training pipeline reaches a mean test
accuracy of 97.96% after selecting each checkpoint on validation
loss. This establishes a working configuration on that task.
The audio examples then ask what happens when the same machinery
has to learn a useful description of sound.

\clearpage

## 2. Choosing the interface to sound

The following choices recur throughout the book. The useful operation
and the cost belong together.

| Representation | Useful operation | Main constraint |
| --- | --- | --- |
| Averaged spectral features or MFCCs | Recognize an instrument from a recording | Averaging removes temporal order; the descriptors do not specify a waveform. |
| Log-magnitude frames | Encode and reconstruct spectral shape; interpolate latent codes | Synthesis needs phase, and independent frames supply little temporal context. |
| Mel-spectrogram patches | Learn local patterns across time and frequency | Mel compression loses spectral detail; waveform reconstruction still needs phase. |
| Synthesiser controls | Prescribe pitch, dynamics, and timbre trajectories | The harmonic-plus-noise renderer restricts the sounds it can express. |
| Symbolic note events | Model relationships among voices | A separate instrument supplies the acoustic sound and articulation. |
| Quantized time-frequency atoms | Edit and generate sparse acoustic events | Quantization, phase agreement, and limited context affect the rendered result. |

The table also suggests an order for development. First decide what
the musician or application needs to change. Then establish what
the representation can reconstruct before training a network.
The STFT round trip, mel backprojection, direct synthesis fit, and
atom-token round trip each expose a different source of error.

A larger network may fit its chosen representation more closely.
Recovering information discarded before training requires a change
to that representation or additional information at rendering time.
For the VAEs, borrowed phase is one such addition. For the synthesis
network, measured pitch and loudness are part of the supplied input.

\clearpage

## 3. What a result lets us claim

The shared-data experiment in [chapter 24](24_differentiable_synthesis.md)
makes these distinctions concrete. On the reserved cello recording,
the VAE's gain-aligned spectral residual is 22.57% and the
network-controlled synthesiser's is 28.77%. On the flute recording
the ordering reverses: 1.99% and 1.55%. These are specific recordings
of known instruments, measured after the application programs'
output processing.

The direct fit of the cello's controls reaches 2.93% on the same
gain-aligned measure. It has access to its target throughout
optimization. That result gives a reason to investigate the shared
network's control mapping and training; it does not establish a
globally optimal fit. The VAE's normalization and observed PCM
clipping also remain part of its result. Removing a scalar level
difference for measurement cannot undo those changes.

Listening adds questions that this residual does not answer.
A softened attack may be unacceptable in a reconstruction and useful
in a timbre transformation. A coherent sustained tone can still have
the wrong spectral envelope. The decision depends on the operation
we intend to perform and the audible feature we need to preserve.

The sequence models require similar care. A lower token loss
establishes better prediction on the declared split. Voice crossing
in the chorales and excessive atom density reveal other aspects of
the generated material. Holding out part of a recording also asks
a different question from holding out a complete recording.

The [experiment record](../book/experiments/README.md) keeps the
v0.2.0 reruns separate from the earlier recorded experiments.
Keeping data identities, splits, settings, source versions, and
outputs together makes these distinctions available to the next
reader.

\clearpage

## 4. Three experiments to take further

**Classify a spectrogram.** Use the convolutional layers to replace
the frame classifier of [chapter 15](15_sound_classifier.md).
Keep the split by recording and compare with the MFCC baseline.
Retain frame offsets when constructing patches so that removed
silence does not become false temporal adjacency. This tests whether
local time-frequency structure helps the chosen classification task.

**Hold out complete recordings for atom prediction.** Extend the
three-sound corpus of [chapter 29](29_atom_transformer.md), fix a split
by recording, and fit the unigram, Markov, and transformer models
on the same training files. Compare token loss, generated event
structure, and rendered examples. This would examine transfer beyond
the within-recording validation used here.

**Investigate articulation in the synthesis model.** Add short notes
and changing articulations to the sustained-note material. Compare
direct control fits with network predictions on the same targets.
If the direct fit already misses an attack, inspect the renderer and
its control time scale. If the direct fit captures it and the network
does not, investigate the conditioning and training. Preserve the
unsuccessful examples alongside the successful ones.

These projects can begin with the existing components. Weft remains
a small CPU laboratory for inspecting arithmetic and experiments.
Real-time audio use adds requirements on allocation, latency, and
state management that the offline examples do not establish.

## 5. Return to the recording

Open the [listening player](audio/index.html) again. Compare the
cello recording, the transform-only reference, the VAE, the direct
fit, and the network-controlled synthesiser. Choose one feature to
follow through all five: the onset of a partial, a movement in pitch,
or the release into silence. The code now gives you a place to look
for each change you hear.

Then return to the flute phrase and the flute-to-cello transition.
Their notes and trajectories were prescribed. What the network
contributes is a learned mapping along those controls. A next musical
experiment can change that gesture deliberately and examine what
the mapping permits.

The useful outcome of this work is the ability to make such decisions
with access to the whole system: the data being measured, the
parameters being learned, and the mechanism that turns their values
into sound.
