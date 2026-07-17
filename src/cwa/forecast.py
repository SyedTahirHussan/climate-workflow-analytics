"""Forecasting a climate index -- the AI-analytics step of the workflow.

Thesis goal (3) is *the connection of AI analytics into the workflow*. Once the
GPU operators and the in-transit reducer have collapsed a 4-D archive into a
compact index time series (e.g. global-mean temperature per month), that series
is the natural handoff point to a predictive model. This module forecasts it.

Two methodological commitments that matter for time series -- and that a
reviewer will look for:

* **No leakage, temporal split.** The test set is the *last* slice of the
  series and the model never sees it during fitting or feature construction.
  Shuffling time-series rows (the default in generic ML tutorials) leaks the
  future into the past and reports fantasy accuracy. We split chronologically.
* **An honest baseline.** A seasonal-naive forecast ("next January looks like
  last January, shifted by the local trend") is genuinely hard to beat on
  seasonal climate data. Reporting the ML model *and* the baseline, and the
  skill score between them, is the difference between "the model works" and "the
  model adds value over the obvious heuristic."

The learner is a gradient-boosted tree on lag/seasonal/rolling features, applied
recursively for multi-step horizons. It is intentionally light and CPU-friendly:
the contribution here is the correct evaluation and the clean integration seam,
not a heavyweight architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def build_features(
    series: np.ndarray, period: int = 12, max_lag: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a supervised (X, y) table from a 1-D series.

    Features per time step ``t`` (all strictly from the past, so no leakage):

    * ``lag_1 .. lag_max_lag`` -- recent values (short-memory dynamics),
    * ``lag_period`` -- the value one seasonal cycle ago (seasonality),
    * ``roll_mean`` / ``roll_std`` over the last ``period`` steps (local level
      and volatility),
    * ``sin``/``cos`` of the seasonal phase (a smooth calendar encoding).

    Rows without enough history (the first ``max(max_lag, period)`` steps) are
    dropped. Returns ``(X, y)`` aligned so ``y[i]`` is the target for ``X[i]``.
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    start = max(max_lag, period)

    rows, targets = [], []
    for t in range(start, n):
        feats = [series[t - lag] for lag in range(1, max_lag + 1)]
        feats.append(series[t - period])  # same phase, previous cycle
        window = series[t - period : t]
        feats.append(window.mean())
        feats.append(window.std())
        phase = 2.0 * np.pi * (t % period) / period
        feats.append(np.sin(phase))
        feats.append(np.cos(phase))
        rows.append(feats)
        targets.append(series[t])

    return np.asarray(rows), np.asarray(targets)


def seasonal_naive_forecast(
    history: np.ndarray, horizon: int, period: int = 12
) -> np.ndarray:
    """Baseline: repeat the last seasonal cycle, plus a local linear trend.

    For horizon step ``h`` the forecast is the observation from one period
    earlier at the matching phase, adjusted by the average per-step change over
    the last cycle. Cheap, transparent, and a strong yardstick on seasonal data.
    """
    history = np.asarray(history, dtype=float)
    if history.size < period + 1:
        raise ValueError("history must be at least one full period + 1 long")

    # Average per-step drift estimated from the most recent cycle.
    recent = history[-(period + 1) :]
    drift = float(np.mean(np.diff(recent)))

    preds = []
    for h in range(1, horizon + 1):
        base = history[-period + (h - 1) % period]  # matching phase last cycle
        preds.append(base + drift * h)
    return np.asarray(preds)


@dataclass
class ForecastReport:
    """Evaluation of a forecast against held-out truth."""

    horizon: int
    model_mae: float
    model_rmse: float
    baseline_mae: float
    baseline_rmse: float
    y_true: np.ndarray
    y_model: np.ndarray
    y_baseline: np.ndarray

    @property
    def skill_score(self) -> float:
        """Fractional MAE improvement of the model over the baseline.

        ``1 - model_mae / baseline_mae``. Positive means the ML model beats the
        seasonal-naive baseline; <= 0 means it does not, which is a result worth
        reporting honestly rather than hiding.
        """
        if self.baseline_mae == 0:
            return 0.0
        return 1.0 - self.model_mae / self.baseline_mae


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def evaluate_forecast(
    series: np.ndarray,
    horizon: int = 12,
    period: int = 12,
    max_lag: int = 12,
    random_state: int = 0,
) -> ForecastReport:
    """Fit on all but the last ``horizon`` steps, forecast them, and score.

    The recursive multi-step scheme feeds each prediction back in as the lag
    input for the next step -- the honest way to forecast ``horizon`` steps when
    only a one-step model is trained, since at inference time future values are
    unknown. Errors therefore compound with horizon, as they should.
    """
    series = np.asarray(series, dtype=float)
    if series.size <= horizon + max(max_lag, period) + 1:
        raise ValueError("series too short for the requested horizon/features")

    # Chronological split: train strictly precedes test.
    train = series[:-horizon]
    y_true = series[-horizon:]

    # Fit the one-step-ahead model on the training portion only.
    X, y = build_features(train, period=period, max_lag=max_lag)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=random_state
    )
    model.fit(X, y)

    # Recursive multi-step forecast, extending a working history as we go.
    working = list(train)
    y_model = []
    for _ in range(horizon):
        arr = np.asarray(working)
        t = arr.size
        feats = [arr[t - lag] for lag in range(1, max_lag + 1)]
        feats.append(arr[t - period])
        window = arr[t - period : t]
        feats.append(window.mean())
        feats.append(window.std())
        phase = 2.0 * np.pi * (t % period) / period
        feats.append(np.sin(phase))
        feats.append(np.cos(phase))
        pred = float(model.predict(np.asarray(feats).reshape(1, -1))[0])
        y_model.append(pred)
        working.append(pred)  # feed the prediction back in
    y_model = np.asarray(y_model)

    y_baseline = seasonal_naive_forecast(train, horizon=horizon, period=period)

    model_mae, model_rmse = _metrics(y_true, y_model)
    base_mae, base_rmse = _metrics(y_true, y_baseline)

    return ForecastReport(
        horizon=horizon,
        model_mae=model_mae,
        model_rmse=model_rmse,
        baseline_mae=base_mae,
        baseline_rmse=base_rmse,
        y_true=y_true,
        y_model=y_model,
        y_baseline=y_baseline,
    )
