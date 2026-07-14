"""Array backend abstraction for transparent CPU/GPU execution.

Thesis goal (1) is *node-local efficient processing via GPUs*. Climate
operators must run on whichever device is fastest for the current data size
without the calling code being rewritten per device. This module gives us a
single seam for that.

Design idea
-----------
NumPy and CuPy expose (almost) the same array API. We therefore select an
"array module" (``xp``) at runtime and write all numerical kernels against it.
The same operator source runs on the host (NumPy) or the device (CuPy):

    xp = get_array_module(Device.GPU)   # -> cupy if available, else numpy
    a  = xp.asarray(host_array)         # host -> device (no-op on CPU)
    m  = xp.mean(a)                     # runs on the selected device

This mirrors how RAPIDS / CuPy code is written in practice and keeps the
operator layer (``operators.py``) device-agnostic. The alternative -- writing
raw CUDA kernels -- is out of scope for a prototype and would couple every
operator to one vendor's toolkit.

CuPy is an *optional* dependency. On a machine without a CUDA device the import
fails and every ``Device.GPU`` request transparently degrades to the CPU path,
with the fallback recorded so callers (and the scheduler) can see it happened.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Optional GPU backend -------------------------------------------------
# Import CuPy lazily and defensively. A missing CUDA runtime raises at import
# time, which we treat as "no GPU on this node" rather than a hard error.
try:  # pragma: no cover - exercised only on GPU nodes
    import cupy as _cupy  # type: ignore

    # Touch the runtime so a broken/absent driver fails here, not mid-kernel.
    _cupy.cuda.runtime.getDeviceCount()
    _CUPY_AVAILABLE = True
except Exception:  # ImportError, CUDARuntimeError, ...
    _cupy = None
    _CUPY_AVAILABLE = False


class Device(enum.Enum):
    """Execution target for a numerical kernel."""

    CPU = "cpu"
    GPU = "gpu"


def gpu_available() -> bool:
    """Return True iff a usable CuPy/CUDA device is present on this node."""
    return _CUPY_AVAILABLE


def get_array_module(device: Device):
    """Return the array module (``numpy`` or ``cupy``) for ``device``.

    Requesting :attr:`Device.GPU` on a host without CuPy silently returns
    NumPy. Silent fallback is intentional: correctness must never depend on the
    presence of a GPU. Whether a fallback occurred is observable via
    :func:`resolve_device`.
    """
    if device is Device.GPU and _CUPY_AVAILABLE:
        return _cupy
    return np


def resolve_device(requested: Device) -> Device:
    """Resolve the *effective* device after applying availability fallback.

    ``resolve_device(Device.GPU)`` returns ``Device.CPU`` when no GPU exists.
    Used for honest logging/metrics: we report what actually ran, not what was
    asked for.
    """
    if requested is Device.GPU and not _CUPY_AVAILABLE:
        return Device.CPU
    return requested


def to_device(array: Any, device: Device):
    """Move ``array`` to ``device``.

    Host->device transfer is explicit here because it is often the dominant
    cost for small arrays -- the exact effect the ML scheduler must learn to
    weigh (see ``scheduler.py``). On the CPU path this is a cheap
    ``np.asarray`` and usually a no-op.
    """
    xp = get_array_module(device)
    return xp.asarray(array)


def to_host(array: Any) -> np.ndarray:
    """Copy ``array`` back to host memory as a NumPy array.

    Accepts NumPy or CuPy input so downstream code (I/O, plotting, sklearn)
    always receives host arrays regardless of where the kernel ran.
    """
    if _CUPY_AVAILABLE and isinstance(array, _cupy.ndarray):  # pragma: no cover
        return _cupy.asnumpy(array)
    return np.asarray(array)


def synchronize(device: Device) -> None:
    """Block until queued device work completes.

    GPU kernels are launched asynchronously; without a sync, a timer would
    measure only the *launch* latency, not the compute. Every benchmark path
    must call this before stopping the clock. No-op on CPU.
    """
    if device is Device.GPU and _CUPY_AVAILABLE:  # pragma: no cover
        _cupy.cuda.runtime.deviceSynchronize()


@dataclass(frozen=True)
class Timing:
    """Result of a single timed execution."""

    device: Device  # the device that actually ran (post-fallback)
    seconds: float  # wall-clock kernel time, transfers excluded unless noted
    n_elements: int  # problem size, for cost-model features


def time_call(fn, *args, device: Device, repeats: int = 3, **kwargs) -> Timing:
    """Time ``fn(*args, **kwargs)`` on ``device`` with correct GPU semantics.

    Reports the *minimum* over ``repeats`` runs. Minimum (rather than mean) is
    the standard choice for micro-benchmarks: it is the run least perturbed by
    OS jitter, other tenants, and turbo/thermal noise, so it best approximates
    the achievable kernel cost that a performance model should target.

    A single warm-up run is discarded to exclude one-off costs (JIT
    compilation, lazy CuPy kernel caching, first-touch page faults).
    """
    effective = resolve_device(requested=device)

    # Warm-up: pay one-time costs (kernel compile, allocator warm-up) once.
    result = fn(*args, **kwargs)
    synchronize(effective)

    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        synchronize(effective)  # ensure async GPU work is finished before stop
        best = min(best, time.perf_counter() - start)

    n = int(getattr(result, "size", 0)) or _infer_size(args)
    return Timing(device=effective, seconds=best, n_elements=n)


def _infer_size(args: tuple) -> int:
    """Best-effort element count from the first array-like positional arg."""
    for a in args:
        size = getattr(a, "size", None)
        if size is not None:
            return int(size)
    return 0
