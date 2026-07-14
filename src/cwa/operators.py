"""Climate/weather field operators, written once, run on CPU or GPU.

The posting names CDO (Climate Data Operators) and Pangeo as the tools this
project must improve on. These operators re-implement a useful subset of the
most common CDO reductions so we can (a) run them on the GPU and (b) drive them
from the in-transit streaming layer. Each operator documents its CDO analogue.

Correctness note -- area weighting
----------------------------------
A latitude/longitude grid does *not* have equal-area cells: a 1-degree cell
near the pole covers far less surface than one at the equator. A naive
``mean(field)`` therefore over-weights the poles and is physically wrong. The
standard fix (what CDO ``fldmean`` does) is to weight each cell by its area,
which on a regular lat-lon grid is proportional to ``cos(latitude)``. We apply
that weighting; getting this right is the difference between a plausible global
mean temperature and a biased one.

All functions take and return arrays from the *same* array module (NumPy or
CuPy). They never move data between host and device -- placement is the
caller's / scheduler's responsibility -- which keeps them pure and cheap to
compose in a streaming pipeline.
"""

from __future__ import annotations

import numpy as np


def latitude_weights(lat_deg, xp=np):
    """Area weights proportional to ``cos(latitude)`` for a regular grid.

    Parameters
    ----------
    lat_deg : array of latitudes in degrees, shape ``(nlat,)``.
    xp : array module the weights should live in (``numpy`` or ``cupy``).

    Returns
    -------
    1-D weight array, shape ``(nlat,)``, clipped at 0 to guard against tiny
    negative ``cos`` values from floating-point error at +/-90 degrees.
    """
    lat = xp.asarray(lat_deg)
    w = xp.cos(xp.deg2rad(lat))
    return xp.clip(w, 0.0, None)


def spatial_mean(field, lat_deg, lat_axis: int = -2, lon_axis: int = -1, xp=np):
    """Area-weighted mean over the spatial (lat, lon) dimensions.

    CDO analogue: ``cdo fldmean``. Given ``field`` shaped ``(..., lat, lon)``
    returns the weighted mean over lat/lon, preserving any leading axes (e.g.
    time). This is the canonical way to collapse a 2-D field into a single
    representative value (a "field mean") such as global-mean temperature.

    The weight broadcasts along longitude because cell area on a regular grid
    depends on latitude only.
    """
    field = xp.asarray(field)
    w_lat = latitude_weights(lat_deg, xp=xp)  # (nlat,)

    # Reshape the 1-D latitude weights so they broadcast against ``field``.
    shape = [1] * field.ndim
    shape[lat_axis] = w_lat.shape[0]
    w = w_lat.reshape(shape)

    # Broadcast to full field shape, then reduce over both spatial axes at once
    # so the normalisation uses exactly the weights that covered valid data.
    w_full = xp.broadcast_to(w, field.shape)
    num = xp.sum(field * w_full, axis=(lat_axis, lon_axis))
    den = xp.sum(w_full, axis=(lat_axis, lon_axis))
    return num / den


def zonal_mean(field, lon_axis: int = -1, xp=np):
    """Unweighted mean along longitude, giving a latitude profile.

    CDO analogue: ``cdo zonmean``. Longitude cells at a fixed latitude have
    equal area, so a plain mean along the longitude axis is correct here -- no
    weighting required. Produces the classic lat-vs-value zonal profile.
    """
    field = xp.asarray(field)
    return xp.mean(field, axis=lon_axis)


def climatology(field, period: int, time_axis: int = 0, xp=np):
    """Mean seasonal cycle over a repeating ``period`` (e.g. 12 months).

    CDO analogue: ``cdo ymonmean`` (for monthly climatology). Folds the time
    axis into ``period`` phase bins and averages each bin across all cycles,
    yielding the expected value for each phase (Jan, Feb, ... for monthly
    data). This is the baseline that anomalies are measured against.

    Requires the length of the time axis to be a whole multiple of ``period``.
    """
    field = xp.asarray(field)
    field = xp.moveaxis(field, time_axis, 0)
    n = field.shape[0]
    if n % period != 0:
        raise ValueError(
            f"time length {n} is not a multiple of period {period}; "
            "align the series before computing a climatology"
        )
    cycles = n // period
    # (cycles, period, ...) -> mean over the cycles axis -> (period, ...)
    folded = field.reshape((cycles, period) + field.shape[1:])
    clim = xp.mean(folded, axis=0)
    return xp.moveaxis(clim, 0, time_axis)


def anomaly(field, period: int, time_axis: int = 0, xp=np):
    """Deviation of each time step from its climatological phase mean.

    CDO analogue: ``cdo ymonsub`` (subtract the monthly climatology). Anomalies
    remove the dominant seasonal cycle so that trends, extremes, and events
    (heatwaves, ENSO signals) become visible. By construction the anomaly of a
    full number of cycles averages to ~0 at every phase -- a property the unit
    tests assert.
    """
    field = xp.asarray(field)
    clim = climatology(field, period=period, time_axis=time_axis, xp=xp)

    field_t = xp.moveaxis(field, time_axis, 0)
    clim_t = xp.moveaxis(clim, time_axis, 0)
    cycles = field_t.shape[0] // period
    # Tile the (period, ...) climatology back over every cycle, then subtract.
    clim_tiled = xp.concatenate([clim_t] * cycles, axis=0)
    result = field_t - clim_tiled
    return xp.moveaxis(result, 0, time_axis)


def running_mean(field, window: int, time_axis: int = 0, xp=np):
    """Centred moving average along the time axis (temporal smoothing).

    CDO analogue: ``cdo runmean,N``. Smooths high-frequency variability to
    expose lower-frequency structure. Implemented via a cumulative-sum sliding
    window -- O(n) rather than the O(n*window) of a naive loop -- which matters
    when the same operator is applied per-chunk to a long streamed series.

    Returns the ``n - window + 1`` valid (fully-populated) windows.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    field = xp.asarray(field)
    field = xp.moveaxis(field, time_axis, 0)
    n = field.shape[0]
    if window > n:
        raise ValueError(f"window {window} exceeds series length {n}")

    # Prefix sums with a leading zero row: window sum = cumsum[i+w] - cumsum[i].
    zero = xp.zeros((1,) + field.shape[1:], dtype=field.dtype)
    csum = xp.concatenate([zero, xp.cumsum(field, axis=0)], axis=0)
    windows = (csum[window:] - csum[:-window]) / window
    return xp.moveaxis(windows, 0, time_axis)


def standardized_anomaly(field, period: int, time_axis: int = 0, eps: float = 1e-12, xp=np):
    """Anomaly divided by the per-phase standard deviation (z-score).

    Standardising by each phase's variability makes anomalies comparable across
    seasons and regions -- a +2 K winter anomaly and a +2 K summer anomaly are
    not equally unusual if summer is more variable. Widely used as an input
    feature for downstream ML on climate fields.
    """
    field = xp.asarray(field)
    anom = anomaly(field, period=period, time_axis=time_axis, xp=xp)

    # Per-phase standard deviation, computed on the folded cycles.
    field_t = xp.moveaxis(field, time_axis, 0)
    cycles = field_t.shape[0] // period
    folded = field_t.reshape((cycles, period) + field_t.shape[1:])
    std = xp.std(folded, axis=0)  # (period, ...)
    std_tiled = xp.concatenate([std] * cycles, axis=0)

    anom_t = xp.moveaxis(anom, time_axis, 0)
    result = anom_t / (std_tiled + eps)
    return xp.moveaxis(result, 0, time_axis)


# Registry used by the scheduler and benchmark harness to refer to operators by
# a stable string key (e.g. when logging which kernel a placement decision was
# made for). Keeping this explicit avoids stringly-typed dispatch scattered
# across the codebase.
OPERATORS = {
    "spatial_mean": spatial_mean,
    "zonal_mean": zonal_mean,
    "anomaly": anomaly,
    "running_mean": running_mean,
    "standardized_anomaly": standardized_anomaly,
}
