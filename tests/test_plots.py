"""Plot functions must produce real PNG files from synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless-safe before any pyplot import

from cwa.forecast import ForecastReport  # noqa: E402
from cwa.plots import plot_cost_model, plot_forecast  # noqa: E402


def _report(horizon=12):
    rng = np.random.default_rng(0)
    truth = np.sin(np.linspace(0, 2 * np.pi, horizon)) + 15.0
    return ForecastReport(
        horizon=horizon,
        model_mae=0.02, model_rmse=0.03,
        baseline_mae=0.03, baseline_rmse=0.04,
        y_true=truth,
        y_model=truth + 0.02 * rng.standard_normal(horizon),
        y_baseline=truth + 0.03 * rng.standard_normal(horizon),
    )


def _benchmark_df():
    sizes = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    rows = []
    for op, (a, tp) in {
        "spatial_mean": (1e-5, 8e8),
        "anomaly": (3e-5, 6e8),
    }.items():
        for device, scale in [("cpu", 1.0), ("gpu", 0.1)]:
            rows.append(pd.DataFrame({
                "op": op, "device": device,
                "n_elements": sizes,
                "seconds": (a + sizes / tp) * scale + (1e-3 if device == "gpu" else 0),
            }))
    return pd.concat(rows, ignore_index=True)


def _assert_png(path):
    assert path.exists()
    assert path.stat().st_size > 1000  # a real rendered image, not a stub
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_forecast_writes_png(tmp_path):
    out = tmp_path / "forecast.png"
    history = 15.0 + np.sin(np.linspace(0, 8 * np.pi, 48))
    plot_forecast(_report(), out, history=history)
    _assert_png(out)


def test_plot_forecast_works_without_history(tmp_path):
    out = tmp_path / "forecast_nohist.png"
    plot_forecast(_report(), out)
    _assert_png(out)


def test_plot_cost_model_writes_png(tmp_path):
    out = tmp_path / "benchmark.png"
    plot_cost_model(_benchmark_df(), out)
    _assert_png(out)
