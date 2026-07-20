"""Forecast correctness: leakage-free features, honest baseline, sound metrics."""

import numpy as np
import pytest

from cwa.forecast import (
    ForecastReport,
    build_features,
    evaluate_forecast,
    seasonal_naive_forecast,
)


def _seasonal_series(n=120, period=12, trend=0.0, noise=0.0, seed=0):
    """A sinusoidal annual cycle, optional linear trend, optional noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cycle = np.sin(2.0 * np.pi * (t % period) / period)
    return cycle + trend * t + noise * rng.standard_normal(n)


def test_seasonal_naive_is_exact_on_pure_cycle():
    # On a noise-free, trend-free periodic series the baseline should be
    # perfect: same phase last cycle IS next year's value, and the drift term
    # vanishes because the diffs over one full period sum to zero.
    series = _seasonal_series(n=60)
    horizon = 12
    preds = seasonal_naive_forecast(series[:-horizon], horizon=horizon)
    np.testing.assert_allclose(preds, series[-horizon:], atol=1e-12)


def test_seasonal_naive_rejects_short_history():
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.zeros(12), horizon=3, period=12)  # needs 13


def test_build_features_shapes():
    series = _seasonal_series(n=60)
    X, y = build_features(series, period=12, max_lag=12)
    start = max(12, 12)
    assert X.shape == (60 - start, 12 + 1 + 2 + 2)  # lags + season lag + roll + phase
    assert y.shape == (60 - start,)
    np.testing.assert_array_equal(y, series[start:])


def test_build_features_uses_only_the_past():
    # Leakage test: corrupting the series from index k onward must not change
    # any feature row whose target lies before k. If it did, features would be
    # peeking at the future.
    series = _seasonal_series(n=80, noise=0.1)
    k = 50
    corrupted = series.copy()
    corrupted[k:] = 1e6

    X_clean, _ = build_features(series)
    X_corr, _ = build_features(corrupted)
    start = 12  # first emitted row targets t = start
    n_unaffected = k - start
    np.testing.assert_array_equal(X_clean[:n_unaffected], X_corr[:n_unaffected])


def test_evaluate_forecast_chronological_split():
    series = _seasonal_series(n=120, trend=0.002, noise=0.05)
    horizon = 12
    report = evaluate_forecast(series, horizon=horizon)
    # The held-out truth must be exactly the LAST `horizon` points — a
    # chronological split, never a shuffle.
    np.testing.assert_array_equal(report.y_true, series[-horizon:])
    assert report.y_model.shape == (horizon,)
    assert report.y_baseline.shape == (horizon,)
    for value in (report.model_mae, report.model_rmse,
                  report.baseline_mae, report.baseline_rmse):
        assert np.isfinite(value) and value >= 0


def test_evaluate_forecast_rejects_short_series():
    # The guard needs size > horizon + max(max_lag, period) + 1 = 25.
    with pytest.raises(ValueError):
        evaluate_forecast(_seasonal_series(n=25), horizon=12)


def test_skill_score_sign_convention():
    def report_with(model_mae, baseline_mae):
        z = np.zeros(1)
        return ForecastReport(
            horizon=1, model_mae=model_mae, model_rmse=0.0,
            baseline_mae=baseline_mae, baseline_rmse=0.0,
            y_true=z, y_model=z, y_baseline=z,
        )

    assert report_with(0.5, 1.0).skill_score == pytest.approx(0.5)  # model wins
    assert report_with(1.0, 0.5).skill_score == pytest.approx(-1.0)  # model loses
    assert report_with(1.0, 0.0).skill_score == 0.0  # guarded division
