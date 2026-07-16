"""Benchmark harness and analytical performance model.

The methods section of the posting lists *modelling of system and application
performance behavior* and *evaluation methods*. This module provides both:

* a reproducible harness that times each operator across a range of problem
  sizes on every available device, and
* a simple, interpretable analytical model fitted to those timings that
  captures the two costs that decide CPU-vs-GPU placement.

Analytical model
----------------
For a data-parallel reduction, wall time is modelled as

    t(n) = fixed_overhead + n / throughput

``fixed_overhead`` lumps together launch/dispatch latency and (on GPU) the
host<->device transfer of a small result; ``throughput`` is the asymptotic
elements/second the device sustains. This is a deliberately minimal, roofline-
flavoured model: two parameters, both physically meaningful, fitted by least
squares. It is not meant to predict every microarchitectural effect -- it is
meant to explain *why* the GPU loses on small fields (its ``fixed_overhead``
dominates) and wins on large ones (its higher ``throughput`` dominates), which
is precisely the trade-off the scheduler must exploit.

The fitted per-device models also generate the crossover size -- the problem
size at which switching devices pays off -- which is the ground truth the
learned scheduler is evaluated against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backend import Device, gpu_available, time_call, to_device
from .operators import OPERATORS


@dataclass(frozen=True)
class GridSpec:
    """A (time, lat, lon) shape plus the number of seasonal phases it holds."""

    n_time: int
    n_lat: int
    n_lon: int
    period: int = 12

    @property
    def n_elements(self) -> int:
        return self.n_time * self.n_lat * self.n_lon


def default_grids() -> list[GridSpec]:
    """A geometric sweep of grid sizes from tiny to ~50M elements.

    Geometric spacing (rather than linear) covers several orders of magnitude
    with few points, which is what a log-scale cost curve needs to be well
    determined at both the overhead-bound and throughput-bound ends.
    """
    return [
        GridSpec(n_time=12, n_lat=16, n_lon=32),  # ~6K
        GridSpec(n_time=12, n_lat=45, n_lon=90),  # ~49K
        GridSpec(n_time=24, n_lat=90, n_lon=180),  # ~389K
        GridSpec(n_time=60, n_lat=90, n_lon=180),  # ~1.0M
        GridSpec(n_time=48, n_lat=180, n_lon=360),  # ~3.1M
        GridSpec(n_time=120, n_lat=180, n_lon=360),  # ~7.8M
        GridSpec(n_time=240, n_lat=360, n_lon=720),  # ~62M
    ]


def _make_field(spec: GridSpec, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random field and its latitude vector for a given grid spec."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(
        (spec.n_time, spec.n_lat, spec.n_lon), dtype=np.float32
    )
    lat = np.linspace(-89.0, 89.0, spec.n_lat).astype(np.float32)
    return field, lat


def _invoke(op_name: str, field, lat, spec: GridSpec, xp):
    """Call operator ``op_name`` with the arguments it expects.

    Operators have slightly different signatures (some need latitude, some need
    the seasonal period); this adapter keeps the benchmark loop uniform.
    """
    op = OPERATORS[op_name]
    if op_name == "spatial_mean":
        return op(field, lat, lat_axis=-2, lon_axis=-1, xp=xp)
    if op_name == "zonal_mean":
        return op(field, lon_axis=-1, xp=xp)
    if op_name in ("anomaly", "standardized_anomaly"):
        return op(field, period=spec.period, time_axis=0, xp=xp)
    if op_name == "running_mean":
        return op(field, window=min(12, spec.n_time), time_axis=0, xp=xp)
    raise KeyError(op_name)


def benchmark(
    op_names: list[str] | None = None,
    grids: list[GridSpec] | None = None,
    devices: list[Device] | None = None,
    repeats: int = 3,
) -> pd.DataFrame:
    """Time each (operator, grid, device) combination.

    Returns a tidy DataFrame with one row per measurement:
    ``[op, device, n_time, n_lat, n_lon, n_elements, bytes, seconds]``.
    This table is both the evaluation artefact and the training set the
    scheduler learns its cost model from.
    """
    op_names = op_names or list(OPERATORS.keys())
    grids = grids or default_grids()
    if devices is None:
        devices = [Device.CPU] + ([Device.GPU] if gpu_available() else [])

    rows = []
    for spec in grids:
        field_host, lat_host = _make_field(spec)
        for device in devices:
            # Transfer once, outside the timed region, so we measure the kernel
            # rather than the (separately modelled) transfer.
            field = to_device(field_host, device)
            lat = to_device(lat_host, device)
            from .backend import get_array_module

            xp = get_array_module(device)
            for op_name in op_names:
                timing = time_call(
                    _invoke,
                    op_name,
                    field,
                    lat,
                    spec,
                    xp,
                    device=device,
                    repeats=repeats,
                )
                rows.append(
                    {
                        "op": op_name,
                        "device": timing.device.value,
                        "n_time": spec.n_time,
                        "n_lat": spec.n_lat,
                        "n_lon": spec.n_lon,
                        "n_elements": spec.n_elements,
                        "bytes": spec.n_elements * field_host.itemsize,
                        "seconds": timing.seconds,
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CostModel:
    """Fitted ``t(n) = fixed_overhead + n / throughput`` for one op+device."""

    op: str
    device: str
    fixed_overhead: float  # seconds; latency floor at n -> 0
    throughput: float  # elements per second (asymptotic slope^-1)
    r2: float  # goodness of fit, for trust in the model

    def predict(self, n_elements: float) -> float:
        return self.fixed_overhead + n_elements / self.throughput


def fit_cost_model(df: pd.DataFrame, op: str, device: str) -> CostModel:
    """Fit ``t = a + b*n`` (a>=0, b>=0) by relative-weighted least squares.

    Two deliberate choices make this fit trustworthy across a size sweep that
    spans four orders of magnitude:

    * **Relative weighting** (weight ``1/t^2``). Ordinary least squares minimises
      *absolute* residuals, so the multi-second large-n points dominate and the
      microsecond small-n points -- which actually determine the overhead term --
      contribute almost nothing. Weighting by ``1/t^2`` minimises *relative*
      error instead, giving every decade equal say. This is standard practice
      when fitting cost/latency curves.
    * **Non-negativity.** Overhead and per-element cost are both physically
      non-negative. If the unconstrained fit returns a negative intercept
      (which happens when the true overhead is below the measurement noise, or
      the curve bends upward at large n from cache/bandwidth effects), we clamp
      it to zero and refit the slope -- the exact KKT solution for this
      two-parameter non-negative problem. A reported overhead of 0 then honestly
      means "not resolvable above the noise floor of these measurements".
    """
    sub = df[(df["op"] == op) & (df["device"] == device)]
    if len(sub) < 2:
        raise ValueError(f"need >=2 points to fit {op}/{device}, got {len(sub)}")

    n = sub["n_elements"].to_numpy(dtype=float)
    t = sub["seconds"].to_numpy(dtype=float)
    w = 1.0 / np.clip(t, 1e-12, None) ** 2  # relative-error weights

    # Weighted normal equations for [a, b].
    sw = np.sum(w)
    swn = np.sum(w * n)
    swnn = np.sum(w * n * n)
    swt = np.sum(w * t)
    swnt = np.sum(w * n * t)
    A = np.array([[sw, swn], [swn, swnn]])
    rhs = np.array([swt, swnt])
    a, b = np.linalg.solve(A, rhs)

    # Enforce non-negativity (KKT): if a<0, pin a=0 and refit b through origin.
    if a < 0:
        a = 0.0
        b = swnt / swnn if swnn > 0 else 0.0
    if b < 0:
        b = 0.0
        a = swt / sw if sw > 0 else 0.0

    pred = a + b * n
    # Weighted R^2 so the goodness measure matches what we optimised.
    wmean = np.sum(w * t) / sw
    ss_res = float(np.sum(w * (t - pred) ** 2))
    ss_tot = float(np.sum(w * (t - wmean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    throughput = 1.0 / b if b > 0 else float("inf")
    return CostModel(
        op=op,
        device=device,
        fixed_overhead=float(a),
        throughput=float(throughput),
        r2=r2,
    )


def crossover_elements(cpu: CostModel, gpu: CostModel) -> float | None:
    """Problem size where GPU becomes faster than CPU, or None if never.

    Solves ``cpu.predict(n) == gpu.predict(n)`` for n. Below the crossover the
    CPU wins (GPU overhead not amortised); above it the GPU wins. Returns None
    when the lines do not cross for positive n (one device dominates
    everywhere).
    """
    # a_c + n/T_c = a_g + n/T_g  ->  n * (1/T_c - 1/T_g) = a_g - a_c
    denom = (1.0 / cpu.throughput) - (1.0 / gpu.throughput)
    if denom == 0:
        return None
    n = (gpu.fixed_overhead - cpu.fixed_overhead) / denom
    return n if n > 0 else None
