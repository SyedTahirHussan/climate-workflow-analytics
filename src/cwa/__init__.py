"""climate-workflow-analytics (cwa).

A compact, GPU-capable prototype for post-processing climate/weather data,
built around the three goals of the HPS "Smart Data Analytics for
Climate/Weather Workflows" PhD topic:

1. node-local GPU processing        -> ``cwa.backend`` + ``cwa.operators``
2. in-transit (streaming) processing -> ``cwa.streaming``
3. AI analytics in the workflow      -> ``cwa.forecast`` + ``cwa.scheduler``

Performance modelling and evaluation live in ``cwa.perfmodel``; the ML-based
heterogeneous scheduler in ``cwa.scheduler``.
"""

from __future__ import annotations

from .backend import Device, gpu_available
from .forecast import ForecastReport, evaluate_forecast
from .operators import OPERATORS, anomaly, running_mean, spatial_mean, zonal_mean
from .perfmodel import CostModel, benchmark, crossover_elements, fit_cost_model
from .scheduler import Placement, SmartScheduler
from .streaming import InTransitProcessor, OnlineMoments, stream_time_slices

__version__ = "0.1.0"

__all__ = [
    "Device",
    "gpu_available",
    "OPERATORS",
    "spatial_mean",
    "zonal_mean",
    "anomaly",
    "running_mean",
    "InTransitProcessor",
    "OnlineMoments",
    "stream_time_slices",
    "benchmark",
    "fit_cost_model",
    "crossover_elements",
    "CostModel",
    "SmartScheduler",
    "Placement",
    "evaluate_forecast",
    "ForecastReport",
    "__version__",
]
