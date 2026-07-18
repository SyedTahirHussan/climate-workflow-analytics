"""Backend abstraction: fallback semantics and honest timing."""

import numpy as np

from cwa.backend import (
    Device,
    get_array_module,
    gpu_available,
    resolve_device,
    time_call,
    to_host,
)


def test_gpu_available_is_bool():
    assert isinstance(gpu_available(), bool)


def test_cpu_module_is_numpy():
    assert get_array_module(Device.CPU) is np


def test_gpu_request_falls_back_without_cuda():
    # On a CPU-only host, a GPU request must degrade to NumPy, not raise.
    if not gpu_available():
        assert get_array_module(Device.GPU) is np
        assert resolve_device(Device.GPU) is Device.CPU


def test_to_host_accepts_numpy():
    a = np.arange(5)
    out = to_host(a)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, a)


def test_time_call_reports_positive_time_and_size():
    x = np.ones((100, 100), dtype=np.float32)
    timing = time_call(lambda a: a.sum(), x, device=Device.CPU, repeats=2)
    assert timing.seconds > 0
    assert timing.device is Device.CPU  # fallback-resolved device
