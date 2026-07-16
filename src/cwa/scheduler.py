"""ML-based scheduling: learn where each operator should run.

This implements the posting's *smart scheduling using Machine Learning to
distribute the workloads efficiently across a heterogeneous landscape*.

Problem
-------
Given an operator and an input size, choose the device (CPU or GPU) that will
finish first. The right choice is size-dependent: GPUs have high throughput but
pay a fixed launch/transfer overhead, so tiny fields are faster on the CPU and
large fields are faster on the GPU. Hard-coding a single size threshold per
operator is brittle -- the crossover moves with the operator, the hardware, and
the data layout. Instead we *learn* a per-device runtime model from measured
timings and place each task by predicted cost.

Approach
--------
For every device we fit a regressor that predicts execution time from
``(operator, size)``. At decision time we predict the runtime on each device and
pick the minimum. This turns placement into a tiny, explainable model-comparison
rather than a wall of if-statements, and adding a third device (say, a second
GPU generation) means training one more model, not editing the policy.

Two modelling choices, both deliberate:

* **Learn in log-time.** Runtimes span several orders of magnitude across the
  size sweep. Regressing raw seconds would let the largest points dominate the
  loss and could predict negative times. Training on ``log(seconds)`` yields a
  well-scaled target and guarantees positive predictions after exponentiation.
* **Log-size feature.** The dominant relationship is roughly linear in problem
  size across decades, so ``log10(n_elements)`` is the natural predictor; the
  operator identity is one-hot encoded so each kernel gets its own cost curve.

The estimator is a gradient-boosted tree ensemble -- it captures the mild
non-linearity of the cost curve without assuming a functional form. On the small
benchmark grids here it is effectively a smooth learned lookup table; in
production the same code trains on thousands of real scheduling traces per
hardware type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .backend import Device


@dataclass(frozen=True)
class Placement:
    """A scheduling decision with the evidence behind it."""

    op: str
    n_elements: int
    device: Device  # chosen device (min predicted runtime)
    predicted_seconds: dict[str, float]  # per-device predicted runtime
    speedup: float  # predicted_slowest / predicted_chosen (>= 1.0)


class SmartScheduler:
    """Learned CPU/GPU placement policy.

    Fit once on a benchmark table (:func:`cwa.perfmodel.benchmark`), then call
    :meth:`choose` per task. The scheduler is device-set agnostic: it trains one
    runtime model per device value present in the training data.
    """

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self._models: dict[str, Pipeline] = {}
        self._ops: list[str] = []
        self._devices: list[str] = []

    # -- feature construction ------------------------------------------------
    @staticmethod
    def _features(op: pd.Series | list, n_elements) -> pd.DataFrame:
        """Assemble the model input frame from operator name(s) and size(s).

        Kept in one place so training and inference cannot drift apart -- a
        classic source of silent ML bugs.
        """
        n = np.asarray(n_elements, dtype=float)
        return pd.DataFrame(
            {
                "op": list(op),
                "log_n": np.log10(np.clip(n, 1.0, None)),
            }
        )

    def _new_pipeline(self) -> Pipeline:
        # One-hot the operator, pass log-size through numerically, feed a small
        # boosted ensemble. ``handle_unknown='ignore'`` keeps inference from
        # crashing if asked about an operator not seen in training (it degrades
        # to the size-only signal instead).
        pre = ColumnTransformer(
            transformers=[
                ("op", OneHotEncoder(handle_unknown="ignore"), ["op"]),
                ("num", "passthrough", ["log_n"]),
            ]
        )
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=2,  # shallow: the cost curve is smooth, avoid overfitting
            learning_rate=0.05,
            random_state=self.random_state,
        )
        return Pipeline([("pre", pre), ("model", model)])

    # -- training ------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> SmartScheduler:
        """Train one runtime model per device from a benchmark DataFrame.

        Expects columns ``op``, ``device``, ``n_elements``, ``seconds``.
        """
        required = {"op", "device", "n_elements", "seconds"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"benchmark frame missing columns: {missing}")

        self._ops = sorted(df["op"].unique())
        self._devices = sorted(df["device"].unique())

        for device in self._devices:
            sub = df[df["device"] == device]
            X = self._features(sub["op"], sub["n_elements"])
            # Train on log-seconds; add a tiny floor so log is always finite.
            y = np.log(np.clip(sub["seconds"].to_numpy(dtype=float), 1e-9, None))
            self._models[device] = self._new_pipeline().fit(X, y)
        return self

    # -- inference -----------------------------------------------------------
    def predict_seconds(self, op: str, n_elements: int) -> dict[str, float]:
        """Predicted runtime (seconds) for ``op`` at ``n_elements`` per device."""
        if not self._models:
            raise RuntimeError("scheduler is not fitted; call fit() first")
        X = self._features([op], [n_elements])
        return {
            device: float(np.exp(model.predict(X)[0]))
            for device, model in self._models.items()
        }

    def choose(self, op: str, n_elements: int) -> Placement:
        """Return the device with the lowest predicted runtime for this task."""
        preds = self.predict_seconds(op, n_elements)
        best_device = min(preds, key=preds.get)
        slowest = max(preds.values())
        speedup = slowest / preds[best_device] if preds[best_device] > 0 else 1.0
        return Placement(
            op=op,
            n_elements=int(n_elements),
            device=Device(best_device),
            predicted_seconds=preds,
            speedup=speedup,
        )

    def crossover(self, op: str, lo: int = 1_000, hi: int = 100_000_000) -> int | None:
        """Smallest size at which the chosen device switches away from CPU.

        Binary-searches the size axis for the operator's learned crossover --
        useful for validating the model against the analytical crossover from
        ``perfmodel`` and for explaining a policy ("this op moves to GPU above
        ~N cells"). Returns None if the choice never changes over ``[lo, hi]``.
        """
        if "cpu" not in self._devices or len(self._devices) < 2:
            return None
        if self.choose(op, lo).device is not Device.CPU:
            return lo  # already non-CPU at the low end
        if self.choose(op, hi).device is Device.CPU:
            return None  # still CPU at the high end -> never switches

        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.choose(op, mid).device is Device.CPU:
                lo = mid
            else:
                hi = mid
        return hi


# ---------------------------------------------------------------------------
# Heterogeneous-hardware demonstration helper.
#
# The sandbox / CI host has no GPU, so real benchmarks contain CPU rows only and
# the placement decision is trivial. To exercise the *scheduling logic* across a
# heterogeneous landscape, we synthesise a GPU timing table from an explicit,
# documented cost model rather than from measurements.
#
# This is a MODEL, not data: it encodes the qualitative GPU behaviour (much
# higher throughput, higher fixed overhead) so the crossover exists and the
# scheduler has something to learn. On a real GPU node this function is not
# used -- ``perfmodel.benchmark([...], devices=[Device.CPU, Device.GPU])``
# returns measured GPU rows and the scheduler trains on those directly.
# ---------------------------------------------------------------------------
def model_gpu_timings(
    cpu_df: pd.DataFrame,
    throughput_speedup: float = 40.0,  # GPU sustains ~40x the elements/s of CPU
    fixed_overhead_s: float = 3.0e-4,  # ~0.3 ms launch + transfer floor
    noise: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a plausible SYNTHETIC GPU timing table shaped like ``cpu_df``.

    Built by taking each CPU measurement's implied per-element cost, dividing it
    by ``throughput_speedup``, and adding a fixed ``fixed_overhead_s`` floor --
    so the GPU is slower on tiny fields (overhead-bound) and faster on large
    ones (throughput-bound), producing a realistic crossover. Multiplicative
    noise avoids a perfectly clean fit.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in cpu_df.iterrows():
        n = float(r["n_elements"])
        # Attribute the CPU time to per-element work (ignoring its small
        # overhead) and speed that portion up on the GPU.
        cpu_per_elem = r["seconds"] / max(n, 1.0)
        gpu_time = fixed_overhead_s + (cpu_per_elem / throughput_speedup) * n
        gpu_time *= 1.0 + rng.normal(0.0, noise)
        row = r.to_dict()
        row["device"] = Device.GPU.value
        row["seconds"] = max(gpu_time, 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)
