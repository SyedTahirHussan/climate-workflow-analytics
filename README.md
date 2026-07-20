# climate-workflow-analytics (`cwa`)

[![CI](https://github.com/SyedTahirHussan/climate-workflow-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/SyedTahirHussan/climate-workflow-analytics/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A compact, tested, GPU-capable prototype for post-processing climate/weather
data with in-transit streaming and machine-learning analytics.

I built this to engage concretely with the HPS PhD topic
[**Smart Data Analytics for Climate/Weather Workflows**](https://hps.vi4io.org/research/open-theses/phd/smarter-workflows)
(Dr. Julian Kunkel / Bryan Lawrence, ESiWACE2, in collaboration with NVIDIA).
The topic identifies three gaps — command-line tools like **CDO** have limited
parallelism, Python stacks like **Pangeo** are not built for on-line
processing, and AI integration is inefficient — and sets three goals. This
repository is a small, honest implementation of those goals so the ideas are
runnable rather than described.

> Scope: a **prototype** that demonstrates the concepts on synthetic data on a
> single node. It is deliberately readable and well-tested rather than
> production-scale. See [`docs/design_rationale.md`](docs/design_rationale.md)
> for what it is, what it is not, and every trade-off behind it.

## How it maps to the topic

| Topic goal / method | Where it lives | What it shows |
| --- | --- | --- |
| **1. Node-local GPU processing** | `backend.py`, `operators.py` | One operator implementation runs on NumPy **or** CuPy via an `xp` array-module seam; GPU-aware timing |
| **2. In-transit (streaming) processing** | `streaming.py` | Slice-by-slice reduction in **O(one slice)** memory, reader overlapped with compute via a bounded queue; numerically stable Welford statistics |
| **3. AI analytics in the workflow** | `forecast.py`, `scheduler.py` | Climate-index forecasting with leakage-free evaluation; a **learned** CPU/GPU placement policy |
| Method: performance modelling + evaluation | `perfmodel.py` | Benchmark harness + analytical `t(n)=overhead+n/throughput` cost model + crossover analysis |
| Method: ML scheduling across heterogeneous hardware | `scheduler.py` | Per-device runtime models; size-dependent placement; learned crossover, cross-checked against the analytical one |
| Domain parity with CDO | `operators.py` | Area-weighted mean (`fldmean`), zonal mean (`zonmean`), anomaly (`ymonsub`), running mean (`runmean`), standardized anomaly |

## Install

CPU-only (works anywhere, including CI):

```bash
git clone https://github.com/syedtahirhussan/climate-workflow-analytics.git
cd climate-workflow-analytics
pip install -e ".[dev]"
```

On a CUDA 12 GPU node, add the GPU extra to enable the CuPy path:

```bash
pip install -e ".[dev,gpu]"    # installs cupy-cuda12x
```

CuPy is an **optional** dependency: if it is absent, every GPU request
transparently falls back to CPU and records that it did so — correctness never
depends on a GPU being present.

## Quick start

```bash
cwa gen                # write a synthetic CF-NetCDF dataset
cwa stream             # in-transit reduce it to a climate index (goal 2)
cwa benchmark          # time operators + fit the performance model (methods)
cwa schedule           # train the ML placement policy, show crossovers (methods)
cwa forecast           # forecast the index vs a seasonal-naive baseline (goal 3)
```

Or use it as a library — see [`examples/`](examples/):

```python
from cwa import InTransitProcessor, evaluate_forecast
from cwa.streaming import spatial_mean_reducer, stream_time_slices
from cwa.backend import Device

proc = InTransitProcessor(spatial_mean_reducer, device=Device.GPU)  # -> CPU if no GPU
result = proc.run(stream_time_slices("cwa_demo.nc", var="tas"))
report = evaluate_forecast(result.series, horizon=12)
print(report.skill_score)   # ML skill vs seasonal-naive baseline
```

## Representative results

Numbers below are from a run of the suite on a CPU-only host (no GPU present);
they are illustrative of behaviour, not a hardware benchmark.

**In-transit streaming** (`cwa stream`, 240 monthly 90×180 slices) processes the
whole series holding **one 64.8 kB slice** at a time, and the streamed index is
**bit-for-bit equal** to a full-array computation (`max |streamed − batch| =
0.0`) — streaming here is exact, not approximate.

**Performance model** (`cwa benchmark`) — analytical cost model per operator
(CPU), fitted by relative-weighted non-negative least squares:

| operator | fixed overhead | throughput | weighted R² |
| --- | ---: | ---: | ---: |
| `zonal_mean` | 7.7 µs | 3117 Melem/s | 0.98 |
| `spatial_mean` | 15.2 µs | 809 Melem/s | 0.90 |
| `anomaly` | 33.9 µs | 604 Melem/s | 0.92 |
| `running_mean` | 2.1 µs | 222 Melem/s | 0.92 |
| `standardized_anomaly` | 46.5 µs | 191 Melem/s | 0.93 |

The ordering is physical: `zonal_mean` (a plain axis reduction) is the fastest
per element; `standardized_anomaly` (anomaly + per-phase std + divide) the
slowest.

**Learned scheduler** (`cwa schedule`, GPU timings *modelled* on this CPU-only
host — clearly labelled in the output). The learned CPU→GPU crossover orders
correctly by per-element cost — the *cheapest* CPU operator needs the *largest*
field before the GPU wins:

| operator | learned CPU→GPU crossover |
| --- | ---: |
| `standardized_anomaly` | ~17 K elements |
| `spatial_mean` | ~137 K elements |
| `zonal_mean` | ~615 K elements |

For `spatial_mean` the **learned** crossover (~137 K) agrees with the
**analytical** crossover (~222 K) to within an order of magnitude — two
independent methods, consistent answer.

**Forecasting** (`cwa forecast`, 12-month horizon, strict temporal split): the
gradient-boosted model reaches MAE 0.018 °C vs the seasonal-naive baseline's
0.027 °C — a **+34% skill score**. The baseline is reported alongside because a
model that cannot beat seasonal-naive on seasonal data has added nothing.

## Tests

```bash
pytest            # 24 tests
```

The tests assert *correctness properties*, not just "it runs":
area-weighting matches a hand computation and differs from a naive mean;
anomalies sum to zero per phase; streamed statistics equal batch statistics;
Welford matches NumPy at large means; the scheduler learns the correct
size-dependent device and a crossover in the right decade.

## Repository layout

```
src/cwa/
  backend.py     CPU/GPU array-module seam + GPU-aware timing
  operators.py   climate reductions (device-agnostic), mapped to CDO
  data.py        synthetic CF-NetCDF generator
  streaming.py   in-transit processor + Welford online statistics
  perfmodel.py   benchmark harness + analytical cost model + crossover
  scheduler.py   ML-based CPU/GPU placement
  forecast.py    climate-index forecasting + honest baseline
  cli.py         `cwa` command-line entry point
examples/        runnable programmatic demos (with correctness cross-checks)
tests/           24 correctness tests
docs/            architecture.md, design_rationale.md
```

## Author

**Syed Tahir Hussan** — Islamabad, Pakistan
[GitHub](https://github.com/syedtahirhussan) ·
[Portfolio](https://syedtahirhussan.github.io/syedtahirhussan) ·
[LinkedIn](https://linkedin.com/in/syedtahirhussan)

Licensed under the MIT License — see [`LICENSE`](LICENSE).
