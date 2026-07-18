# Design rationale

Short notes on the decisions that shaped this prototype — what was chosen, why,
and what was traded away. The point of the repository is to demonstrate
reasoning about a systems + ML + climate problem, so the reasoning is written
down rather than left implicit.

## 1. One kernel, two devices (the `xp` seam)

**Decision.** Write every operator against an abstract array module (`xp`) that
resolves to NumPy or CuPy at runtime, rather than writing CUDA kernels.

**Why.** The thesis needs *node-local GPU processing*, but a prototype's value
is in the concepts, not in hand-tuned kernels. NumPy/CuPy API compatibility lets
one implementation target both devices and keeps the code readable — which is
explicitly what the application says it evaluates. CuPy already dispatches to
cuBLAS/cuDNN/Thrust under the hood, so the GPU path is not naive.

**Trade-off.** A vendor-neutral, array-level approach leaves ~10–30% on the
table versus fused custom kernels for memory-bound reductions, and it inherits
CuPy's kernel-launch overhead per operation. For fields large enough to want the
GPU at all, that overhead amortises; for tiny fields the scheduler routes to CPU
anyway. If profiling later shows a hot operator, it can be replaced by a fused
`cupy.RawKernel` behind the same interface without changing callers.

## 2. In-transit streaming over load-then-reduce

**Decision.** Process one time slice at a time through a bounded
producer/consumer queue, keeping O(one slice) memory, instead of the
Pangeo-style "open the dataset, then reduce" pattern.

**Why.** Goal 2 asks for processing that can run *concurrently with
applications*. Streaming caps memory independent of series length (a 40-year run
uses the same RAM as a 1-year run) and overlaps I/O with compute. The bounded
queue provides backpressure, so if compute lags, the reader blocks rather than
buffering the whole archive.

**Trade-off.** Streaming forbids operators that need the whole series in memory
at once (e.g. a global quantile). Those either use an online/approximate
algorithm (t-digest for quantiles) or a two-pass variant. For the reductions
here — means, anomalies, running means — an exact single-pass or online form
exists, and the tests confirm the streamed result equals the batch result to
floating-point.

## 3. Welford for online statistics

**Decision.** Accumulate mean/variance with Welford's algorithm, not
`sum(x)`/`sum(x²)`.

**Why.** Temperatures carried in Kelvin have a large mean relative to their
variance; the textbook sum-of-squares formula subtracts two large nearly-equal
numbers and loses most of the significant digits. Welford updates the mean and
the centred sum-of-squares incrementally and stays accurate. It also has an
exact parallel-merge rule, which is what a distributed reduction needs.

**Trade-off.** Slightly more arithmetic per update than the naive sums —
irrelevant next to the I/O and reduction cost, and bought back many times over
in numerical trust.

## 4. A two-parameter analytical cost model

**Decision.** Model runtime as `t(n) = fixed_overhead + n / throughput` and fit
it by relative-weighted, non-negativity-constrained least squares.

**Why.** The scheduler's whole premise is a size-dependent CPU/GPU trade-off,
and this two-parameter model is the smallest thing that *explains* it: GPUs lose
on small fields (their `fixed_overhead` dominates) and win on large ones (their
`throughput` dominates). Both parameters are physically interpretable, and the
model yields a closed-form crossover to validate the learned scheduler against.
Relative weighting stops the multi-second large-n points from swamping the
microsecond small-n points that pin the overhead term; the non-negativity
constraint keeps overhead physical (a reported 0 means "below the measurement
noise floor").

**Trade-off.** The linear model ignores cache/bandwidth regime changes, so its
weighted R² sits around 0.9 rather than ~1.0 — deliberately. It is an
explanatory model for placement, not a high-fidelity performance predictor. The
learned scheduler (next section) is what actually makes decisions; the
analytical model is the sanity check.

## 5. Learned scheduling, not a hard-coded threshold

**Decision.** Learn a per-device runtime model and place each task by predicted
cost, rather than hard-coding "use GPU above N elements".

**Why.** Goal/method: *ML scheduling across a heterogeneous landscape*. A single
threshold is wrong the moment the operator, the GPU generation, the dtype, or
the data layout changes. A learned model absorbs all of those as features, and
adding a device means training one more model, not editing a policy tree. Log-
time targets and a log-size feature keep the regression well-conditioned across
orders of magnitude.

**Trade-off.** ML placement needs training data (benchmark traces) and can
mispredict off-distribution. Mitigations for a production version: keep the
analytical model as a guard rail, fall back to it when the learned prediction is
low-confidence, and periodically re-benchmark to track hardware/driver drift.

**Honesty note.** The CI/sandbox host has no GPU, so real benchmarks are
CPU-only. To exercise the heterogeneous decision, a *documented synthetic GPU
cost model* generates the GPU timing rows (`scheduler.model_gpu_timings`),
clearly labelled everywhere it appears. On a real GPU node the benchmark returns
measured GPU rows and the synthetic path is unused. The learned crossovers order
correctly by per-element cost (the cheapest CPU operator crosses to GPU at the
largest size), and agree with the analytical crossover to within an order of
magnitude — evidence the machinery is sound, to be re-validated on real
hardware.

## 6. Forecasting: correct evaluation over a fancy model

**Decision.** A gradient-boosted model on lag/seasonal/rolling features, scored
with a strict chronological split against a seasonal-naive baseline.

**Why.** Goal 3 is *connecting AI analytics into the workflow*; the defensible
contribution at prototype stage is the *integration seam* (streamed index →
forecaster) and *honest evaluation*, not a bespoke deep architecture. Time-
series ML lives or dies on evaluation discipline: shuffling rows leaks the
future and inflates accuracy, and a model that cannot beat seasonal-naive on
seasonal data has added nothing. Reporting both, plus the skill score, is the
honest result (here: +34% MAE improvement over baseline on the synthetic index).

**Trade-off.** Recursive multi-step forecasting compounds error with horizon
(shown in the per-month table) and a tree model cannot extrapolate a trend
beyond its training range. A production version would add direct multi-horizon
models or a hybrid trend + residual-ML approach, and would validate on real
reanalysis indices rather than a synthetic series.

## What this repository is and is not

**Is:** a readable, tested demonstration of the concepts the topic targets —
device-agnostic operators, in-transit streaming, a performance model, a learned
scheduler, and an AI-analytics seam — with the reasoning and trade-offs made
explicit.

**Is not:** a production HPC tool. It uses synthetic data, single-node
execution, and a modelled (not measured) GPU on CPU-only hosts. Those are the
first things the PhD work would replace: real archives, MPI/multi-GPU execution,
measured hardware, and rigorous evaluation on domain benchmarks.
