"""Goal 3 -- AI analytics: forecast the climate index vs a strong baseline.

Derives a global-mean temperature index by streaming the archive through the
GPU-capable spatial-mean reducer, then forecasts the next 12 months with a
gradient-boosted model and scores it against a seasonal-naive baseline using a
strict, leakage-free temporal split.

Run:  python examples/03_forecasting.py
"""

from pathlib import Path

from cwa.backend import Device
from cwa.data import write_netcdf
from cwa.forecast import evaluate_forecast
from cwa.streaming import InTransitProcessor, spatial_mean_reducer, stream_time_slices

PATH = Path("cwa_demo.nc")

if __name__ == "__main__":
    if not PATH.exists():
        write_netcdf(PATH, n_time=240, chunk_time=1)

    # Pipeline handoff: streaming reducer -> compact index -> forecaster.
    proc = InTransitProcessor(spatial_mean_reducer, device=Device.CPU)
    index = proc.run(stream_time_slices(str(PATH), var="tas")).series

    report = evaluate_forecast(index, horizon=12, period=12)

    print("12-month forecast of the global-mean temperature index\n")
    print(f"  gradient-boosted model : MAE={report.model_mae:.4f}  "
          f"RMSE={report.model_rmse:.4f} degC")
    print(f"  seasonal-naive baseline: MAE={report.baseline_mae:.4f}  "
          f"RMSE={report.baseline_rmse:.4f} degC")
    print(f"  skill score            : {report.skill_score:+.1%} vs baseline")

    print("\n  month  truth    model    baseline")
    for i, (yt, ym, yb) in enumerate(
        zip(report.y_true, report.y_model, report.y_baseline, strict=True), start=1
    ):
        print(f"  {i:>5}  {yt:6.3f}   {ym:6.3f}   {yb:6.3f}")
