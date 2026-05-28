# 17 — Making training practical

Up to the audio VAE we never had to think about speed. IRIS trains in a
blink; the MNIST classifier in a minute or two. You could change a
hyperparameter, rerun, and see the result before your coffee cooled.

The audio VAE broke that. Its first layer is 2049 wide, and the SOL
corpus is hundreds of thousands of frames — so a single epoch does on the
order of 10^12 multiply-adds, and a naive run can take many minutes per
epoch. And it only gets heavier from here: convolutional layers, the
ConvVAE, and the GAN are all more expensive than what we've built so far.
This is the point where a few optimization tricks stop being optional.

The good news is that we don't need anything exotic, and nothing here
changes what the library *is*. Three kinds of move: compute the same work
faster, do less work, and make long runs safe.

## 1. Where the time goes

Almost all of it is in one place: the matrix multiply in `Dense`,
`Z = W X`. Everything else — activations, the loss, the Adam update — is
linear in the data and cheap by comparison. The first layer dominates
because it's the widest: `W` is `256 x 2049`, so each batch is a
`256 x 2049` by `2049 x B` multiply, far larger than any later layer's.
Optimising the matmul optimises training. So that single function is
where every trick below is aimed.

## 2. Compute the same work faster

**Vectorization (free).** Compiling at `-O3` lets the compiler turn the
inner loop of the matmul into SIMD instructions — one machine
instruction multiplying several numbers at once (NEON on Apple Silicon,
AVX on x86). Our matmul is written in `i-k-j` order precisely so the
innermost loop walks contiguously through memory, which is both
cache-friendly and easy for the compiler to vectorise. This costs one
flag and typically buys a few times speedup.

**Threads (cheap, big).** The matmul is *embarrassingly parallel*: each
output row of `Z` depends only on one row of `W` and all of `X`, and
different output rows never touch the same memory. So we split the result
rows into contiguous blocks and compute the blocks on separate
`std::thread`s — no locks, no shared writes, no synchronisation beyond a
join at the end. With as many threads as cores, this scales nearly
linearly: an eight-core machine does the big matmuls roughly eight times
faster.

Two details matter. First, threads have a launch cost, so it's only worth
parallelising when there's enough arithmetic to amortise it — the matmul
runs serially below a work threshold (small matmuls, like the ones over
the 32-D latent, just run on one thread). Second, we use `std::thread`
rather than OpenMP deliberately: `std::thread` is part of the C++
standard library, so it needs no compiler extension and no external
runtime, and it compiles unchanged under Apple Clang, GNU g++, and
anything else. That keeps the library's "standard library only, no
dependencies" rule intact — arguably more so than OpenMP, which is a
compiler add-on that Apple's toolchain doesn't even ship.

## 3. Do less work

The cheapest computation is the one you skip. Audio is extremely
redundant frame to frame: with a 4096-sample window and a 2048 hop,
neighbouring frames overlap by half and, on a sustained note, look almost
identical. Training on every single frame spends most of its effort
re-learning the same spectrum. So `train_audio_vae` can **subsample** —
train on every Nth frame. With hundreds of thousands of frames, taking
every third or fourth costs almost nothing in quality and cuts epoch time
proportionally.

One implementation note worth its own line: we subsample the *index
list*, not the data matrix. The cached feature matrix is already several
gigabytes; copying a subsampled version would briefly double that. Instead
we build a list of the column indices we want and select mini-batches from
the original matrix through that list — no large copy, same memory
footprint.

(Subsampling and a smaller dataset also argue for slightly more epochs,
since each epoch sees fewer distinct examples — but given how alike the
dropped frames are, it's nearly free in practice.)

## 4. Make long runs safe

This one isn't about speed; it's about not losing work. A long training
run that only saves at the very end is fragile — a crash, a closed laptop,
or a `Ctrl-C` at minute 90 throws everything away. So `train_audio_vae`
**checkpoints**: every few epochs it writes the current encoder and
decoder to the same `<prefix>.enc`/`.dec` files (serialization, note 16,
makes this a one-liner). Kill the run whenever you like and the latest
weights are already on disk. It also lets you *listen* to an intermediate
model — load the epoch-10 checkpoint in `vae_remap`, decide whether more
training is actually improving things, and stop early if not.

## 5. What we deliberately did not do

There are two bigger hammers we left in the drawer, on purpose.

A tuned **BLAS** library (OpenBLAS, Apple's Accelerate) would beat our
hand-written matmul by another large factor — they use cache blocking and
hand-written assembly we're not going to reproduce. But linking one breaks
the dependency-free promise, and the point of this library is to *read*
the matmul and understand it, not to call into a black box. So a BLAS
stays an optional, off-by-default path at most.

A **GPU** would be faster still, by one to two orders of magnitude. But it
is a different kind of project, not a flag: the data has to live in device
memory, every operation needs a kernel, and the whole mental model shifts
from "read the loop" to "manage a device." That's the *how to make neural
nets fast on accelerators* course, which is genuinely separate from the
*how do neural nets work* course this library is. We leave it out by
choice.

The honest summary: the tricks here are engineering, not magic. They don't
change the number of multiply-adds the math requires — vectorization and
threads just do them faster, subsampling does fewer of them, and
checkpointing protects the run. Stacked together they're enough to keep
CPU training of everything in this library, up through the GAN, bearable
on a laptop. Knowing which lever to pull, and when reaching for a BLAS or
a GPU is worth the cost, is itself part of doing machine learning.
