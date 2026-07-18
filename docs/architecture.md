# Architecture

This prototype is organised as a thin, composable pipeline. Each stage maps to
one goal or method from the HPS *Smart Data Analytics for Climate/Weather
Workflows* topic, and the stages are decoupled through small, explicit
interfaces so that any one can be swapped (e.g. a real GPU backend, an MPI
executor, a different forecaster) without touching the others.

```
          ┌──────────────┐   time slices   ┌────────────────────┐
 NetCDF ──▶│  io / stream │────────────────▶│ InTransitProcessor │──▶ climate index (series)
 (CF)     │  (reader thr)│  bounded queue  │   + OnlineMoments  │        │
          └──────────────┘   backpressure  └─────────┬──────────┘        │
                                                     │ per-slice reduce  │
                                            ┌────────▼──────────┐        │
                                            │    operators      │        │
                                            │  (device-agnostic)│        │
                                            └────────┬──────────┘        │
                                                     │ xp = numpy|cupy   │
                                            ┌────────▼──────────┐        │
                                            │     backend       │        │
                                            │  CPU/GPU + timing │        │
                                            └───────────────────┘        │
                                                                         ▼
   ┌──────────────┐   timings    ┌─────────────────┐          ┌──────────────────┐
   │  perfmodel   │─────────────▶│  SmartScheduler │          │    forecast      │
   │  benchmark   │  (train set) │  (ML placement) │          │  GBM vs baseline │
   │  cost model  │◀─── validate │  crossover      │          │  temporal split  │
   └──────────────┘   crossover  └─────────────────┘          └──────────────────┘
```

## Modules

| Module | Responsibility | Maps to |
| --- | --- | --- |
| `backend.py` | Select NumPy/CuPy array module; host↔device transfer; correct GPU-aware timing | Goal 1 (node-local GPU) |
| `operators.py` | Climate reductions (area-weighted mean, zonal mean, anomaly, running mean, standardized anomaly), written once, run on either device | Goal 1; CDO parity |
| `data.py` | Synthetic CF-compliant NetCDF generator (seasonal cycle, lat gradient, trend, correlated noise) | Reproducible demos/tests |
| `streaming.py` | In-transit reader/consumer with bounded-queue backpressure; Welford online statistics | Goal 2 (in-transit) |
| `perfmodel.py` | Benchmark harness; analytical `t(n)=overhead+n/throughput` cost model; CPU/GPU crossover | Methods: performance modelling, evaluation |
| `scheduler.py` | Learned per-device runtime models; size-dependent placement; learned crossover | Methods: ML scheduling across heterogeneous hardware |
| `forecast.py` | Climate-index forecasting; leakage-free temporal split; seasonal-naive baseline; skill score | Goal 3 (AI analytics) |
| `cli.py` | `cwa {gen,stream,benchmark,schedule,forecast}` — living documentation of the flow | — |

## Key interfaces (the seams that make it scale)

- **`xp` array module.** Operators take an array module argument, so the same
  kernel runs on host or device. Adding a new backend is adding a branch in
  `get_array_module`, not rewriting operators.
- **Reducer signature `(slice_2d, coords, xp) -> scalar`.** Any per-slice
  reduction plugs into the streaming processor. The processor does not know
  which operator it runs.
- **Source iterator `(index, slice_2d, coords)`.** The streaming processor
  consumes any iterator with this shape — a file reader today, a live
  simulation feed or a network stream tomorrow — so online and offline use the
  same code.
- **Benchmark DataFrame `[op, device, n_elements, seconds, ...]`.** The single
  contract between measurement (`perfmodel`) and learning (`scheduler`). Real
  GPU rows and synthetic GPU rows are interchangeable at this boundary.

## Scale-out path (single node → cluster)

The single-node structure is chosen so scaling out is a change of *executor*,
not a rewrite:

1. **Node-local:** one reader thread + one compute stream (this prototype).
2. **Multi-GPU node:** one `InTransitProcessor` per GPU; the scheduler assigns
   slices/operators to devices by predicted cost.
3. **Cluster / in-transit:** one reader+reducer per MPI rank co-located with the
   simulation ranks; ranks reduce their local subdomain and a final
   `OnlineMoments`-style merge combines partial results (Welford has an exact
   parallel-merge formula, so the distributed reduction stays numerically
   stable).

The GIL limits the single-node threaded overlap to the I/O phase; the cluster
tier sidesteps it entirely by using processes/ranks. Keeping the reducer and
source as small interfaces is what makes that transition mechanical.
