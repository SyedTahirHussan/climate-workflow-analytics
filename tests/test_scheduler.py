"""Scheduler correctness: it must learn the size-dependent device choice."""

import numpy as np
import pandas as pd
import pytest

from cwa.backend import Device
from cwa.scheduler import SmartScheduler


def _two_device_dataset(crossover=1_000_000):
    """Synthetic timings for one op with a known CPU/GPU crossover.

    CPU: low overhead, low throughput.  GPU: high overhead, high throughput.
    Constructed so the two cost lines cross near ``crossover`` elements.
    """
    sizes = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])

    # CPU: t = 1e-6 + n / 1e8   (10 us floor is negligible; ~100 Melem/s)
    cpu_t = 1e-6 + sizes / 1e8
    # GPU: t = 1e-2 + n / 1e10  (10 ms floor; ~10 Gelem/s)
    gpu_t = 1e-2 + sizes / 1e10

    rows = []
    for n, tc, tg in zip(sizes, cpu_t, gpu_t, strict=True):
        rows.append({"op": "spatial_mean", "device": "cpu", "n_elements": int(n), "seconds": tc})
        rows.append({"op": "spatial_mean", "device": "gpu", "n_elements": int(n), "seconds": tg})
    return pd.DataFrame(rows)


def test_scheduler_prefers_cpu_when_small():
    sched = SmartScheduler().fit(_two_device_dataset())
    p = sched.choose("spatial_mean", 10_000)
    assert p.device is Device.CPU  # tiny field: GPU overhead not amortised


def test_scheduler_prefers_gpu_when_large():
    sched = SmartScheduler().fit(_two_device_dataset())
    p = sched.choose("spatial_mean", 100_000_000)
    assert p.device is Device.GPU  # huge field: GPU throughput dominates


def test_learned_crossover_is_in_the_right_decade():
    # The analytical crossover of the two lines above is ~1.01e6 elements.
    # The learned crossover should land within an order of magnitude of it.
    sched = SmartScheduler().fit(_two_device_dataset())
    x = sched.crossover("spatial_mean", lo=1_000, hi=100_000_000)
    assert x is not None
    assert 1e5 <= x <= 1e7


def test_speedup_is_at_least_one():
    sched = SmartScheduler().fit(_two_device_dataset())
    p = sched.choose("spatial_mean", 50_000_000)
    assert p.speedup >= 1.0  # chosen device is never slower than the slowest


def test_fit_rejects_missing_columns():
    bad = pd.DataFrame({"op": ["x"], "device": ["cpu"], "seconds": [0.1]})  # no n_elements
    with pytest.raises(ValueError):
        SmartScheduler().fit(bad)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SmartScheduler().predict_seconds("spatial_mean", 1000)
