"""Operator correctness: the domain math must be right, not just runnable."""

import numpy as np
import pytest

from cwa.operators import (
    anomaly,
    climatology,
    latitude_weights,
    running_mean,
    spatial_mean,
    standardized_anomaly,
    zonal_mean,
)


@pytest.fixture
def grid():
    lat = np.linspace(-89.0, 89.0, 45)
    lon = np.linspace(0.0, 358.0, 90)
    return lat, lon


def test_spatial_mean_of_constant_is_constant(grid):
    # A uniform field must average to its constant regardless of weighting.
    lat, lon = grid
    field = np.full((len(lat), len(lon)), 7.5, dtype=np.float32)
    result = spatial_mean(field, lat, lat_axis=0, lon_axis=1)
    assert np.isclose(float(result), 7.5, atol=1e-4)


def test_area_weighting_matches_manual_and_differs_from_naive(grid):
    # Use lat**2 as the field: symmetric about the equator (so it isn't trivially
    # cancelled like a linear-in-lat field), but the poles carry large values on
    # small-area cells -- exactly the case where cos(lat) weighting matters.
    lat, lon = grid
    profile = lat**2
    field = np.broadcast_to(profile[:, None], (len(lat), len(lon))).astype(np.float64)

    weighted = float(spatial_mean(field, lat, lat_axis=0, lon_axis=1))
    naive = float(field.mean())

    # Manual cos(lat)-weighted latitude mean as ground truth.
    w = np.clip(np.cos(np.deg2rad(lat)), 0, None)
    manual = float(np.sum(profile * w) / np.sum(w))

    assert np.isclose(weighted, manual, atol=1e-6)
    # Down-weighting the poles (large lat**2) lowers the mean vs a flat average.
    assert weighted < naive
    assert not np.isclose(weighted, naive, rtol=1e-2)


def test_latitude_weights_nonneg_and_peak_at_equator():
    lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    w = latitude_weights(lat)
    assert np.all(w >= 0)
    assert w[2] == pytest.approx(w.max())  # equator carries the most area


def test_anomaly_sums_to_zero_per_phase():
    # Over whole cycles, each phase's anomalies must cancel by construction.
    period, cycles = 12, 5
    rng = np.random.default_rng(0)
    series = rng.standard_normal((period * cycles, 4, 4)).astype(np.float64)

    anom = anomaly(series, period=period, time_axis=0)
    folded = anom.reshape((cycles, period, 4, 4))
    per_phase_mean = folded.mean(axis=0)  # mean over cycles for each phase
    np.testing.assert_allclose(per_phase_mean, 0.0, atol=1e-12)


def test_climatology_recovers_known_cycle():
    # Deterministic seasonal signal repeated with no noise -> climatology == cycle.
    period = 12
    one_cycle = np.arange(period, dtype=np.float64)
    series = np.tile(one_cycle, 4)[:, None, None]  # (48, 1, 1)
    clim = climatology(series, period=period, time_axis=0).ravel()
    np.testing.assert_allclose(clim, one_cycle, atol=1e-12)


def test_running_mean_matches_manual_and_length():
    x = np.arange(10, dtype=np.float64)[:, None]  # (10, 1)
    w = 3
    out = running_mean(x, window=w, time_axis=0).ravel()
    expected = np.convolve(x.ravel(), np.ones(w) / w, mode="valid")
    assert out.size == x.shape[0] - w + 1
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_zonal_mean_shape(grid):
    lat, lon = grid
    field = np.random.default_rng(1).standard_normal((6, len(lat), len(lon)))
    out = zonal_mean(field, lon_axis=-1)
    assert out.shape == (6, len(lat))  # longitude collapsed


def test_standardized_anomaly_unit_scale():
    # After standardising, per-phase std of the anomalies should be ~1.
    period, cycles = 12, 8
    rng = np.random.default_rng(2)
    series = (rng.standard_normal((period * cycles, 3, 3)) * 5.0 + 100.0)
    z = standardized_anomaly(series, period=period, time_axis=0)
    folded = z.reshape((cycles, period, 3, 3))
    per_phase_std = folded.std(axis=0)
    np.testing.assert_allclose(per_phase_std, 1.0, atol=1e-6)


def test_climatology_rejects_misaligned_length():
    series = np.zeros((13, 2, 2))  # not a multiple of 12
    with pytest.raises(ValueError):
        climatology(series, period=12)
