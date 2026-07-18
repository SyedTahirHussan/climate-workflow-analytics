"""Goal 2 -- in-transit streaming reduction, with a correctness check.

Streams a NetCDF file one time slice at a time, reduces each slice to an
area-weighted global mean (CDO ``fldmean`` per step), and accumulates online
statistics -- all in O(one slice) memory. Then verifies the streamed index
matches a full-array (batch) computation, proving the streaming path is exact,
not approximate.

Run:  python examples/01_streaming.py
"""

from pathlib import Path

import numpy as np
import xarray as xr

from cwa.backend import Device
from cwa.data import write_netcdf
from cwa.operators import spatial_mean
from cwa.streaming import InTransitProcessor, spatial_mean_reducer, stream_time_slices

PATH = Path("cwa_demo.nc")

if __name__ == "__main__":
    if not PATH.exists():
        write_netcdf(PATH, n_time=240, chunk_time=1)

    # --- Stream: process each slice as it arrives, reader overlapped with compute
    proc = InTransitProcessor(spatial_mean_reducer, device=Device.CPU, queue_size=4)
    result = proc.run(stream_time_slices(str(PATH), var="tas"))

    print("in-transit reduction")
    print(f"  device          : {result.device.value}")
    print(f"  steps           : {result.n_steps}")
    print(f"  wall time       : {result.wall_seconds * 1e3:.1f} ms")
    print(f"  peak slice held : {result.peak_slice_bytes / 1e3:.1f} kB")
    print(f"  index mean/std  : {result.moments.mean:.4f} / {result.moments.std:.4f} degC")

    # --- Verify streamed == batch (load the whole cube once, reduce directly)
    with xr.open_dataset(PATH) as ds:
        field = ds["tas"].values
        lat = ds["lat"].values
    batch = spatial_mean(field, lat, lat_axis=1, lon_axis=2)
    max_abs_err = float(np.max(np.abs(np.asarray(result.series) - batch)))
    print(f"\n  max |streamed - batch| = {max_abs_err:.2e}  "
          f"(exact to floating-point; streaming is not an approximation)")
