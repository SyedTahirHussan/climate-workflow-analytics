"""In-transit streaming: process climate data as it arrives, not all at once.

Thesis goal (2) is *scalable processing of massive data volumes that can run
concurrently with applications (in-transit processing)*. The Pangeo-style
pattern loads a dataset (lazily, but conceptually as a whole) and then reduces
it. In-transit instead consumes each time step as soon as it is available --
exactly as a running simulation emits output -- and folds it into a running
result. Two consequences matter:

1. **Bounded memory.** Peak memory is one time slice plus a small accumulator,
   independent of series length. A 40-year run streams in the same footprint as
   a 1-year run.
2. **Overlap.** Reading/decompressing the next slice happens while the current
   slice is being reduced, hiding I/O latency behind compute (and vice versa).

Implementation
--------------
A single-producer / single-consumer pipeline over a *bounded* queue. The bound
provides backpressure: if compute falls behind I/O, the reader blocks instead of
buffering the whole dataset in RAM -- the property that keeps the footprint
constant.

Honest concurrency caveat: Python threads share the GIL, so this overlaps the
I/O phase (NetCDF read + decompress, which release the GIL) with compute, but
does not give two CPU-bound threads true parallelism. That is the right model
for the single-node prototype; the production design scales this across nodes
with one reader/reducer per rank (MPI) or per GPU, where the same producer/
consumer structure maps onto real parallel hardware. Keeping the seam explicit
here is what makes that scale-out a change of executor, not a rewrite.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .backend import Device, resolve_device, to_device, to_host


def stream_time_slices(
    path: str, var: str = "tas"
) -> Iterator[tuple[int, np.ndarray, dict]]:
    """Yield ``(index, slice_2d, coords)`` one time step at a time, lazily.

    Opens the file without materialising the full variable (xarray keeps it
    lazy) and loads exactly one 2-D field per iteration. This is the read
    pattern that a chunked-by-time NetCDF file (see ``data.write_netcdf``)
    serves cheaply, and it caps reader memory at a single slice.
    """
    with xr.open_dataset(path) as ds:
        da = ds[var]
        lat = np.asarray(ds["lat"].values)
        lon = np.asarray(ds["lon"].values)
        n_time = da.sizes["time"]
        for t in range(n_time):
            # ``.values`` here forces I/O for just this slice.
            slice_2d = np.asarray(da.isel(time=t).values)
            yield t, slice_2d, {"lat": lat, "lon": lon}


class OnlineMoments:
    """Streaming mean/variance via Welford's algorithm.

    A single-pass, numerically stable estimator of mean and variance. The naive
    "sum of x and sum of x^2" formula catastrophically loses precision when the
    mean is large relative to the variance (true for temperatures in Kelvin);
    Welford avoids that by updating the mean and the sum-of-squared-deviations
    incrementally. Correctness of streaming statistics is exactly the kind of
    detail that separates a robust reducer from a demo -- so it is unit-tested
    against a full-array NumPy computation.
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations from the running mean

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance (0 until at least one value is seen)."""
        return self._m2 / self.count if self.count > 0 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


@dataclass
class StreamResult:
    """Outcome of an in-transit run."""

    series: list[float] = field(default_factory=list)  # reduced value per step
    moments: OnlineMoments = field(default_factory=OnlineMoments)
    n_steps: int = 0
    wall_seconds: float = 0.0
    device: Device = Device.CPU
    peak_slice_bytes: int = 0  # largest single slice held -> memory footprint


class InTransitProcessor:
    """Reduce a streamed field to a per-step scalar with I/O/compute overlap.

    Parameters
    ----------
    reducer : callable ``(slice_2d, coords, xp) -> float``
        Maps one 2-D field to a scalar (e.g. an area-weighted spatial mean).
        Runs on the selected device via the array module ``xp``.
    device : which backend the reducer runs on.
    queue_size : bound on in-flight slices; the backpressure knob.
    """

    def __init__(
        self,
        reducer: Callable,
        device: Device = Device.CPU,
        queue_size: int = 4,
    ) -> None:
        self.reducer = reducer
        self.device = resolve_device(device)
        self.queue_size = max(1, queue_size)

    def run(self, source: Iterator[tuple[int, np.ndarray, dict]]) -> StreamResult:
        """Consume ``source`` and return the reduced series + statistics.

        ``source`` is any iterator of ``(index, slice_2d, coords)`` -- a file
        (:func:`stream_time_slices`) or a live producer -- so the same reducer
        works offline on an archive or online on simulation output.
        """
        from .backend import get_array_module

        xp = get_array_module(self.device)
        result = StreamResult(device=self.device)

        # Bounded hand-off queue between the reader thread and this consumer.
        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        _SENTINEL = object()  # marks end-of-stream

        def producer() -> None:
            # Runs in a background thread: pulls slices and enqueues them.
            # ``q.put`` blocks when the queue is full -> backpressure -> the
            # reader waits for the consumer instead of buffering everything.
            for item in source:
                q.put(item)
            q.put(_SENTINEL)

        reader = threading.Thread(target=producer, name="cwa-reader", daemon=True)

        start = time.perf_counter()
        reader.start()
        while True:
            item = q.get()  # blocks until a slice is ready: I/O overlaps compute
            if item is _SENTINEL:
                break
            _, slice_2d, coords = item
            result.peak_slice_bytes = max(result.peak_slice_bytes, slice_2d.nbytes)

            # Move this slice to the device and reduce it. Host<->device
            # transfer is per-slice and small, which is why fine-grained
            # streaming favours the CPU for tiny fields (see the scheduler).
            device_slice = to_device(slice_2d, self.device)
            value = float(to_host(self.reducer(device_slice, coords, xp)))

            result.series.append(value)
            result.moments.update(value)
            result.n_steps += 1

        reader.join()
        result.wall_seconds = time.perf_counter() - start
        return result


# --- Ready-made reducers ---------------------------------------------------
# Small closures adapting the pure operators in ``operators.py`` to the
# ``(slice_2d, coords, xp)`` reducer signature the processor expects.


def spatial_mean_reducer(slice_2d, coords, xp):
    """Area-weighted global mean of one 2-D field (CDO ``fldmean`` per step)."""
    from .operators import spatial_mean

    return spatial_mean(slice_2d, coords["lat"], lat_axis=0, lon_axis=1, xp=xp)


def max_reducer(slice_2d, coords, xp):
    """Global maximum of one field -- e.g. tracking a per-step hot extreme."""
    return xp.max(xp.asarray(slice_2d))
