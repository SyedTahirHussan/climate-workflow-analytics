"""Performance model correctness: the fit must recover known cost parameters."""

import numpy as np
import pandas as pd
import pytest

from cwa.backend import Device
from cwa.perfmodel import (
    CostModel,
    GridSpec,
    benchmark,
    crossover_elements,
    fit_cost_model,
)


def _timings(overhead, throughput, op="spatial_mean", device="cpu"):
    """Noise-free synthetic timings following t(n) = overhead + n/throughput."""
    sizes = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    return pd.DataFrame(
        {
            "op": op,
            "device": device,
            "n_elements": sizes,
            "seconds": overhead + sizes / throughput,
        }
    )


def test_fit_recovers_known_parameters():
    m = fit_cost_model(_timings(overhead=1e-5, throughput=1e9), "spatial_mean", "cpu")
    assert m.fixed_overhead == pytest.approx(1e-5, rel=1e-6)
    assert m.throughput == pytest.approx(1e9, rel=1e-6)
    assert m.r2 == pytest.approx(1.0, abs=1e-9)
    # predict() must reproduce the generating law.
    assert m.predict(1e6) == pytest.approx(1e-5 + 1e6 / 1e9, rel=1e-6)


def test_fit_clamps_negative_overhead_to_zero():
    # Two points whose exact line has a NEGATIVE intercept: physically
    # impossible, so the fit must clamp overhead to 0 and refit the slope.
    df = pd.DataFrame(
        {
            "op": "spatial_mean",
            "device": "cpu",
            "n_elements": [1e6, 2e6],
            "seconds": [1.0, 2.5],  # line through these: t = -0.5 + 1.5e-6 * n
        }
    )
    m = fit_cost_model(df, "spatial_mean", "cpu")
    assert m.fixed_overhead == 0.0
    assert np.isfinite(m.throughput) and m.throughput > 0


def test_fit_requires_two_points():
    df = _timings(1e-5, 1e9).iloc[:1]
    with pytest.raises(ValueError):
        fit_cost_model(df, "spatial_mean", "cpu")


def test_crossover_matches_hand_solution():
    # CPU: t = n/1e8.  GPU: t = 1e-3 + n/1e9.
    # Equal when n(1e-8 - 1e-9) = 1e-3  ->  n = 1e-3 / 9e-9.
    cpu = CostModel(op="x", device="cpu", fixed_overhead=0.0, throughput=1e8, r2=1.0)
    gpu = CostModel(op="x", device="gpu", fixed_overhead=1e-3, throughput=1e9, r2=1.0)
    assert crossover_elements(cpu, gpu) == pytest.approx(1e-3 / 9e-9, rel=1e-9)


def test_crossover_none_when_one_device_dominates():
    cpu = CostModel(op="x", device="cpu", fixed_overhead=1e-4, throughput=1e8, r2=1.0)
    # GPU better on BOTH axes: the lines never cross for positive n.
    gpu = CostModel(op="x", device="gpu", fixed_overhead=1e-5, throughput=1e9, r2=1.0)
    assert crossover_elements(cpu, gpu) is None
    # Identical throughput: parallel lines, no crossover either.
    gpu2 = CostModel(op="x", device="gpu", fixed_overhead=1e-3, throughput=1e8, r2=1.0)
    assert crossover_elements(cpu, gpu2) is None


def test_benchmark_produces_tidy_table():
    # One tiny grid, one operator, CPU only: fast, but exercises the real path.
    df = benchmark(
        op_names=["zonal_mean"],
        grids=[GridSpec(n_time=4, n_lat=8, n_lon=16)],
        devices=[Device.CPU],
        repeats=1,
    )
    assert list(df.columns) == [
        "op", "device", "n_time", "n_lat", "n_lon", "n_elements", "bytes", "seconds",
    ]
    assert len(df) == 1
    assert df.loc[0, "n_elements"] == 4 * 8 * 16
    assert df.loc[0, "seconds"] > 0
