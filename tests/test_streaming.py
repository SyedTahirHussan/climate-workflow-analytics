"""Streaming correctness: online stats and in-transit == batch equivalence."""

import numpy as np

from cwa.backend import Device
from cwa.operators import spatial_mean
from cwa.streaming import (
    InTransitProcessor,
    OnlineMoments,
    spatial_mean_reducer,
)


def test_online_moments_match_numpy():
    # Welford's streaming estimate must equal a full-array computation.
    rng = np.random.default_rng(0)
    data = rng.normal(500.0, 3.0, size=1000)  # large mean stresses stability
    m = OnlineMoments()
    for x in data:
        m.update(float(x))
    assert np.isclose(m.mean, data.mean(), atol=1e-9)
    assert np.isclose(m.variance, data.var(), atol=1e-6)


def test_online_moments_empty():
    m = OnlineMoments()
    assert m.count == 0
    assert m.variance == 0.0  # defined, not a division-by-zero


def _fake_source(field, lat, lon):
    """Yield (t, slice, coords) tuples from an in-memory array."""
    for t in range(field.shape[0]):
        yield t, field[t], {"lat": lat, "lon": lon}


def test_in_transit_equals_batch_spatial_mean():
    # The streamed per-step index must match computing it on the full cube.
    rng = np.random.default_rng(1)
    n_time, n_lat, n_lon = 24, 30, 60
    field = rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float64)
    lat = np.linspace(-89, 89, n_lat)
    lon = np.linspace(0, 358, n_lon)

    proc = InTransitProcessor(spatial_mean_reducer, device=Device.CPU, queue_size=3)
    result = proc.run(_fake_source(field, lat, lon))

    batch = spatial_mean(field, lat, lat_axis=1, lon_axis=2)  # (n_time,)
    np.testing.assert_allclose(np.asarray(result.series), batch, atol=1e-10)
    assert result.n_steps == n_time


def test_peak_memory_is_single_slice():
    # Footprint must be O(one slice), independent of series length.
    rng = np.random.default_rng(2)
    n_time, n_lat, n_lon = 50, 20, 40
    field = rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32)
    lat = np.linspace(-89, 89, n_lat)
    lon = np.linspace(0, 358, n_lon)

    proc = InTransitProcessor(spatial_mean_reducer, device=Device.CPU)
    result = proc.run(_fake_source(field, lat, lon))

    one_slice_bytes = field[0].nbytes
    assert result.peak_slice_bytes == one_slice_bytes
