"""Generate a small, realistic, CF-compliant NetCDF dataset for demos/tests.

Real climate archives are terabytes and gated; a self-contained demo needs data
it can create locally in a second. This module synthesises a temperature-like
field on a regular lat-lon grid with the structure that makes the operators and
the forecaster meaningful:

* a **seasonal cycle** (so ``anomaly`` / ``climatology`` have something to
  remove),
* a **latitudinal gradient** -- warm equator, cold poles (so area weighting in
  ``spatial_mean`` actually changes the answer),
* a slow **warming trend** (so the forecaster has signal to learn), and
* spatially-correlated **noise** (so nothing is trivially perfect).

The file follows CF conventions (coordinate variables with ``units`` and
``standard_name``, a CF ``units``/``calendar`` time axis). CF compliance is what
lets standard tooling -- xarray, CDO, Panoply -- open the file without
hand-holding, and signals that the pipeline speaks the domain's data format
rather than an ad-hoc array dump.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def make_dataset(
    n_time: int = 240,  # 20 years of monthly steps -> exercises climatology
    n_lat: int = 90,
    n_lon: int = 180,
    period: int = 12,  # months per year; matches the seasonal cycle
    trend_per_decade: float = 0.2,  # K/decade global warming signal
    seed: int = 42,
) -> xr.Dataset:
    """Build an in-memory :class:`xarray.Dataset` of surface temperature.

    Deterministic given ``seed`` so tests and benchmarks are reproducible --
    reproducibility being an explicit requirement for research software.
    """
    rng = np.random.default_rng(seed)

    lat = np.linspace(-89.0, 89.0, n_lat)
    lon = np.linspace(0.0, 358.0, n_lon)
    months = np.arange(n_time)

    # --- Deterministic physical structure ---------------------------------
    # Meridional (latitudinal) profile: ~+30 C at equator, ~-25 C at poles.
    lat_profile = 30.0 * np.cos(np.deg2rad(lat)) - 25.0 * np.sin(np.deg2rad(lat)) ** 2
    # Seasonal cycle, phase-flipped across the equator (opposite hemispheres).
    phase = 2.0 * np.pi * (months % period) / period
    seasonal = np.outer(np.sin(phase), np.sin(np.deg2rad(lat)))  # (time, lat)
    seasonal *= 12.0  # amplitude in K
    # Linear warming trend applied uniformly.
    trend = trend_per_decade * (months / period / 10.0)  # (time,)

    # Assemble the mean field: (time, lat, lon), broadcasting across longitude.
    base = lat_profile[None, :, None]  # (1, lat, 1)
    seas = seasonal[:, :, None]  # (time, lat, 1)
    trnd = trend[:, None, None]  # (time, 1, 1)
    mean_field = base + seas + trnd
    mean_field = np.broadcast_to(mean_field, (n_time, n_lat, n_lon))

    # --- Spatially-correlated noise ---------------------------------------
    # White noise looks nothing like weather; a light zonal smoothing gives the
    # field local spatial correlation without pulling in heavy dependencies.
    noise = rng.normal(0.0, 2.0, size=(n_time, n_lat, n_lon))
    kernel = np.array([0.25, 0.5, 0.25])
    noise = (
        np.roll(noise, 1, axis=2) * kernel[0]
        + noise * kernel[1]
        + np.roll(noise, -1, axis=2) * kernel[2]
    )

    temp = (mean_field + noise).astype("float32")

    ds = xr.Dataset(
        data_vars={
            "tas": (
                ("time", "lat", "lon"),
                temp,
                {
                    "standard_name": "air_temperature",
                    "long_name": "near-surface air temperature",
                    "units": "degC",
                },
            )
        },
        coords={
            "time": (
                "time",
                months,
                {"units": "days since 2000-01-01", "calendar": "360_day"},
            ),
            "lat": (
                "lat",
                lat.astype("float32"),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                "lon",
                lon.astype("float32"),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "title": "Synthetic surface temperature for climate-workflow-analytics demos",
            "Conventions": "CF-1.8",
            "source": "cwa.data.make_dataset (synthetic; not real observations)",
        },
    )
    return ds


def write_netcdf(path: str | Path, *, chunk_time: int | None = None, **kwargs) -> Path:
    """Generate a dataset and write it to a NetCDF4 file at ``path``.

    ``chunk_time`` sets the NetCDF internal chunk length along time. Chunking
    the file to match how it will be read (one time slice at a time in the
    streaming layer) lets the reader pull a slice without decompressing the
    whole variable -- the storage-side analogue of the in-transit access
    pattern this project targets.
    """
    path = Path(path)
    ds = make_dataset(**kwargs)

    encoding = {"tas": {"zlib": True, "complevel": 4}}
    if chunk_time is not None:
        n_lat = ds.sizes["lat"]
        n_lon = ds.sizes["lon"]
        encoding["tas"]["chunksizes"] = (chunk_time, n_lat, n_lon)

    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path
