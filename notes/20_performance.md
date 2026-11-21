# 20 — Making training practical

Up to the audio VAE we never had to think about speed. IRIS trains
almost instantly; the MNIST classifier took a minute or two before this
chapter's changes, and takes seconds after them. We could change a
hyperparameter, rerun, and see the result with almost no wait.

The audio VAE is different. Its first layer is 2049 wide, and the
TinySOL corpus is about 200,000 frames, so a single epoch does on the
order of 10^12 multiply-adds, and a naive run can take many minutes per
epoch. The models that follow are heavier still: convolutional layers,
the ConvVAE and the transformer are all more expensive than what we've
built so far. From here on, some optimization is necessary.

We don't need anything exotic, and nothing here changes what the
library is. The techniques either compute the same work faster, do
less work, or make long runs safe.

## 1. Where the time goes

Almost all of it goes to the matrix multiply in `Dense`, `Z = W X`.
In this dense VAE, activations, the loss, and the Adam update are cheap
by comparison. The first encoder layer and the last
decoder layer dominate because they are the widest: each multiplies a
`256 x 2049` (or `2049 x 256`) matrix by the batch, far more work than
any layer in between. Optimising the matmul optimises training, so
every technique below targets that function.

## 2. Compute the same work faster

**Vectorization (free).** Compiling at `-O3` lets the compiler turn the
inner loop of the matmul into SIMD instructions, where one machine
instruction multiplies several numbers at once (NEON on Apple Silicon,
AVX on x86). Our matmul is written in `i-k-j` order so that the
innermost loop walks contiguously through memory, which uses the cache
well and can help the compiler vectorise. The effect of `-O3` needs to
be measured for the compiler, scalar type, and matrix shapes in use.

**Threads (cheap, large gain).** The matmul is *embarrassingly
parallel*: each output row of `Z` depends only on one row of `W` and
all of `X`, and different output rows never touch the same memory. We
split the result rows into contiguous blocks and compute the blocks on
separate `std::thread`s, with no locks, no shared writes, and no
synchronisation beyond a join at the end. Speedup depends on matrix
shape, memory bandwidth, available cores, and thread-launch overhead.
Eight cores do not imply an eightfold gain.

Threads have a launch cost, so parallelising is only worthwhile when
there is enough arithmetic to amortise it. The matmul runs serially
below a work threshold (small matmuls, like the ones over the 32-D
latent, run on one thread). We also use `std::thread` rather than
OpenMP deliberately: `std::thread` is part of the C++ standard
library, so it needs no compiler extension and no external runtime,
and it is supported by the toolchains used for this project. This keeps
the default implementation within the C++ standard library.

## 3. Do less work

Audio is extremely redundant frame to frame: with a 4096-sample window
and a 2048 hop, neighbouring frames overlap by half and, on a
sustained note, look almost identical. Training on every frame spends
most of its effort learning the same spectrum again. So
`train_audio_vae` can **subsample**, that is, train on every Nth
frame. Taking every third or fourth reduces the number of examples per
epoch. Its quality cost must be checked on held-out material, especially
for attacks and other brief events that differ from sustained frames.

We subsample the index list, not the data matrix. The cached feature
matrix is already about 1.6 GB; copying a subsampled version would
briefly double that. Instead we build a list of the column indices we
want and select mini-batches from the original matrix through that
list. There is no large copy and the memory footprint is unchanged.

An epoch over a subsample contains fewer updates, so compare runs by
both elapsed time and the number of examples processed. Equal epoch
counts do not imply equal training budgets.

## 4. Make long runs safe

This technique is about not losing work, not about speed. A long
training run that only saves at the very end is fragile: a crash, a
closed laptop, or a `Ctrl-C` at minute 90 loses everything. So
`train_audio_vae` **checkpoints**: every few epochs it writes the
current encoder and decoder to the same `<prefix>.enc`/`.dec` files
(with the serialization of chapter 19 this takes one line). The run
retains the most recently completed weight save. Checkpoints also let
you listen to an intermediate model: load the
epoch-10 checkpoint in `vae_remap`, decide whether more training is
improving the result, and stop early if not.

These are weight snapshots. They do not store Adam's moments, the
training position, or random-generator state, and loading them is not
an exact continuation of a run. The two files are written separately
to the same paths, so interruption during a save can leave an incomplete
or mismatched pair. A resumable checkpoint format and atomic replacement
would require additional implementation.

## 5. An optional faster backend: BLAS

There is a larger speedup that we make available but leave off by
default. A tuned **BLAS** library (OpenBLAS, Intel MKL, or Apple's
Accelerate) implements matrix multiply with cache blocking and
hand-written assembly that we are not going to reproduce, and for
large matmuls it can outperform our loop. The gain depends on backend,
hardware, and matrix shape, so benchmark the actual training workload.
`Matrix::operator*` therefore
has an optional BLAS path, enabled at compile time
with `make BLAS=1`.

The design keeps it isolated from the rest of the library:

- All BLAS use is in one place, `operator*`, guarded by
  `#ifdef WEFT_USE_BLAS`. Nothing else in the codebase refers to it,
  and the default build neither includes nor links any BLAS, so the
  "standard library only" rule holds for anyone who doesn't ask for it.
- Because `operator*` is a member of `Matrix`, it accesses the
  row-major `data_` buffer directly and passes it to
  `cblas_sgemm` / `cblas_dgemm`. The BLAS path needs nothing from
  outside the class.
- It dispatches with `if constexpr` on the scalar type: `float` and
  `double` go to BLAS, any other type uses the portable threaded loop.
  The feature is therefore type-safe, and unsupported types still
  work.
- The build is OS-aware: on macOS it links Apple's Accelerate framework,
  which ships with the system (nothing to install); elsewhere it links
  OpenBLAS.

This is how we think an optional dependency should be added to a
library like this: in a single isolated place, off by default, so that
a user who wants speed can switch it on at no cost to the rest of the
code or to other users. We wrote our own matmul in order to understand
it. Once it is understood, delegating it to a tuned kernel for
production-scale runs is a reasonable, deliberate choice rather than a
hidden default.

## 6. What we leave out: GPU

A **GPU** can accelerate sufficiently large workloads, but transfer
costs and small matrix shapes can erase the benefit. It would require a
different implementation: the data has to live in
device memory, every operation needs a kernel, and the programmer
manages a device instead of reading a loop. That is the subject of how
to make neural nets fast on accelerators, which is separate from the
subject of this library, how neural nets work. We leave it out by
choice.

The techniques here don't change the number of multiply-adds the math
requires. Vectorization and threads do them faster, an optional BLAS
does them with a better kernel, subsampling does fewer of them, and
checkpointing protects the run. Together they are enough to keep CPU
training of everything in this library, up through the transformer,
practical at the small model and dataset sizes used here.

## 7. Recording a timing result

Keep the command, source revision, compiler and flags, scalar type,
matrix backend, machine, and dataset selection with a timing. Report
wall-clock time and the work performed, such as examples processed or
training steps. Repeat short measurements to distinguish a change in
the implementation from ordinary timing variation. The times elsewhere
in the book describe particular runs, not hardware-independent targets.


## Further reading {#sec-20-performance-further-reading}

The BLAS interface that the optional backend calls is
[@lawson1979blas]. How a tuned matrix product achieves its speed, with
blocking for each level of cache, is described in [@goto2008anatomy].
