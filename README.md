# SIMD Convolution Reverb Generator

**A single-core study in SIMD and instruction-level parallelism.**

This project builds a convolution reverb kernel in C++ and then makes it fast —
one step at a time. Starting from a plain scalar loop, it climbs through five
optimization "rungs" and reaches a **28× speedup** on a single core, using only
SIMD and instruction-level parallelism. No threads, no GPU.

Each rung fixes exactly *one* bottleneck in the rung before it, so every speedup
number can be traced to a single, nameable cause instead of a pile of mixed
changes.

---

## What it does

Convolving a dry audio signal with a room *impulse response* produces
reverberation — the sound of that recording played back inside the room. Each
output sample is a dot product: a long multiply-accumulate, which is the ideal
candidate for SIMD (Single Instruction, Multiple Data).

The signal is floating point throughout, because audio samples and reverb
coefficients are fractional real numbers, not integers.

### Two setup tricks

Before any optimization, two one-time preparations keep the inner loop clean:

- **The impulse response is reversed.** Convolution naturally strides *backward*
  through the signal; reversing the kernel turns that into a contiguous
  *forward* walk, which is what vector loads prefer.
- **The signal is zero-padded on both sides.** This lets the inner loop read
  every sample it needs without a single bounds check — the edge reads land
  safely on padding instead of running off the array.

Both are done outside the timed region, so they cost nothing in the benchmark.

---

## The five rungs

### Rung 1 — Scalar baseline

A plain, unoptimized nested loop that processes one multiply-and-add at a time.
It plays two roles: the **correctness reference** every faster version is checked
against, and the **denominator** every speedup is measured from. It is compiled
with the same optimization settings as the fast kernels, so the comparison stays
fair.

### Rung 2 — Auto-vectorized

The *same loop* as Rung 1, with one change: fast floating-point math is enabled.
Compilers normally refuse to vectorize a floating-point sum on their own, because
floating-point addition is not perfectly associative — reordering the additions
can change the last few bits through rounding. Allowing that reordering frees the
compiler to add several numbers at once on its vector units. This rung shows how
much speed is available essentially **for free**.

### Rung 3 — Hand-written vectors, one accumulator

The first deliberate SIMD version. A 256-bit vector register holds eight
32-bit floats, so one fused multiply-add (FMA) processes eight taps at once.
It's correct and genuinely eight-wide — but it uses a single accumulator, so
every FMA has to wait for the previous one to finish before it can start. Since
an FMA takes several cycles, the whole computation is forced into one serial
chain and spends most of its time waiting. This is called being
**latency-bound**: the hardware can do far more, but the code has serialized it.

### Rung 4 — Multiple accumulators

The fix for Rung 3: keep several independent accumulators. Because they don't
depend on one another, a new FMA can start while earlier ones are still
finishing, keeping the execution units busy instead of idle. This moves the
kernel from latency-bound toward **throughput-bound**.

It then hits a hard ceiling. Every FMA needs *two* memory loads (one impulse
tap, one signal sample), and the CPU has only *two* load ports. Once both are
saturated, throughput can't climb further, and adding still more accumulators
stops helping. The loads have become the limit.

### Rung 5 — Register blocking

The final rung breaks the load ceiling by changing *what the vector lanes hold*.
Instead of packing eight taps of one output into a register, it packs eight
consecutive **outputs**. Each impulse coefficient is then broadcast once and
reused across many output blocks. This shifts the balance from two loads per FMA
toward one, pushing past the load-port wall. As a bonus, each finished output
lands directly in its own lane, so the expensive step of summing across lanes
(needed in Rungs 3 and 4) disappears entirely.

The number of output blocks is capped by the sixteen available vector registers,
so it grows to about eight before performance regresses.

---

## Results

Benchmarked on **~1 second of 48 kHz audio** (48,000 samples) convolved with a
**~0.1 second impulse response** (4,800 taps). Single core, median of 11 runs.
Higher GFLOP/s is better.

| Implementation        | Rung | Time (ms) | GFLOP/s   | Speedup    |
|-----------------------|:----:|----------:|----------:|:----------:|
| scalar (baseline)     |  1   |    236.70 |      2.14 | 1.00×      |
| autovec               |  2   |     34.16 |     14.84 | 6.93×      |
| avx2, 1 accumulator   |  3   |     39.11 |     12.96 | 6.05×      |
| avx2, 2 accumulators  |  4   |     19.29 |     26.28 | 12.27×     |
| avx2, 4 accumulators  |  4   |     19.35 |     26.19 | 12.23×     |
| avx2, 8 accumulators  |  4   |     17.47 |     29.02 | 13.55×     |
| avx2, 16 accumulators |  4   |     20.34 |     24.92 | 11.64×     |
| block, R=2            |  5   |     19.71 |     25.71 | 12.01×     |
| block, R=4            |  5   |      9.90 |     51.22 | 23.92×     |
| **block, R=8**        |  5   |  **8.38** | **60.45** | **28.23×** |

All ten implementations agreed on the hand-checkable sanity test
`[1,2,3] * [1,1] = [1,3,5,3]` and matched the scalar baseline within tolerance.

### What the numbers show

- **Auto-vectorization alone gave 6.9× for free** — a large win before any hand
  tuning.
- **The naive hand-written SIMD (Rung 3) was actually *slower* than the
  compiler's auto-vectorized version** (6.05× vs 6.93×). This is a direct
  measurement of how much the single-accumulator latency chain costs — both do
  comparable work, but Rung 3 serializes it.
- **More accumulators helped only up to a point.** Eight accumulators peaked at
  13.55×; sixteen was *slower*, confirming that once the two load ports saturate,
  extra accumulators just add overhead.
- **Register blocking broke the load-port wall**, more than doubling the best
  Rung 4 result to reach **60.45 GFLOP/s** and an overall **28.23× speedup** —
  all on a single core.
- **Accuracy improved with more accumulators**, because splitting the sum into
  several partial totals is numerically gentler than one long running sum.

---

## Benchmark methodology

- **Median of 11 runs.** Timing noise is one-sided — the OS can only make a run
  *slower* by stealing the core, never faster — so the median discards that slow
  tail and reflects true steady-state performance.
- **Untimed warm-up run.** Absorbs one-time costs (cold caches, page faults)
  before measurement begins.
- **Monotonic clock.** Immune to system time adjustments mid-measurement.
- **Two correctness checks.** An *accuracy check* against a slow but trustworthy
  double-precision reference, and an *agreement check* confirming every rung
  matches the scalar baseline. Both pass within a tolerance that absorbs ordinary
  floating-point rounding.

---

## Building and running

Build and run with `make run`. The project compiles at a single optimization
level so the scalar baseline is measured on equal footing with the kernels it's
compared against.

**Requirements:** a CPU supporting AVX2 and FMA (any x86 chip from ~2013
onward), a GCC-compatible C++ compiler, and C++17.

---

## Why single-threaded?

Deliberately. The goal is to see how far *one core* can go using only vector
width and instruction-level parallelism, before threads enter the picture. The
impulse response here (~18.75 KB) fits entirely in L1 cache, so the compute-side
lever — register blocking — is what matters. For a much longer reverb that spills
out of cache, the next step would be tiling the tap range to keep a working
segment cache-resident: the memory-side half of the same idea, and the natural
continuation of this study.

---

## Author

**Bhargava Bhatkurse**
