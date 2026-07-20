"""Command-line interface tying the pipeline stages together.

Subcommands mirror the thesis goals so the CLI doubles as living documentation:

    cwa gen        -> write a synthetic CF-NetCDF dataset
    cwa stream     -> in-transit reduce it to a climate index (goal 2)
    cwa benchmark  -> time operators and fit the performance model (methods)
    cwa schedule   -> train the ML placement policy and show crossovers (methods)
    cwa forecast   -> forecast the index vs a seasonal-naive baseline (goal 3)

Run ``cwa --help`` or ``python -m cwa.cli --help`` for options.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from . import __version__
from .backend import Device, gpu_available
from .data import write_netcdf
from .forecast import evaluate_forecast
from .perfmodel import benchmark, fit_cost_model
from .scheduler import SmartScheduler, model_gpu_timings
from .streaming import InTransitProcessor, spatial_mean_reducer, stream_time_slices


def _default_path() -> Path:
    return Path(tempfile.gettempdir()) / "cwa_demo.nc"


def cmd_gen(args: argparse.Namespace) -> None:
    path = write_netcdf(
        args.path, n_time=args.n_time, chunk_time=1  # chunk by time for streaming
    )
    size_mb = Path(path).stat().st_size / 1e6
    print(f"wrote {path}  ({size_mb:.1f} MB, {args.n_time} monthly steps)")


def cmd_stream(args: argparse.Namespace) -> None:
    if not Path(args.path).exists():
        write_netcdf(args.path, n_time=args.n_time, chunk_time=1)
    device = Device.GPU if args.gpu else Device.CPU
    proc = InTransitProcessor(spatial_mean_reducer, device=device, queue_size=4)
    result = proc.run(stream_time_slices(args.path, var="tas"))

    print(f"in-transit spatial-mean reduction on {result.device.value}")
    print(f"  steps processed : {result.n_steps}")
    print(f"  wall time       : {result.wall_seconds * 1e3:.1f} ms")
    print(f"  peak slice held : {result.peak_slice_bytes / 1e3:.1f} kB "
          f"(memory is O(one slice), not O(series))")
    print(f"  index mean/std  : {result.moments.mean:.3f} / {result.moments.std:.3f} degC")
    print(f"  first 6 values  : {[round(v, 2) for v in result.series[:6]]}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    df = benchmark(repeats=args.repeats)
    print(f"benchmarked {df['op'].nunique()} operators x "
          f"{df['n_elements'].nunique()} sizes on device(s): "
          f"{sorted(df['device'].unique())}\n")
    # Fit and report the analytical cost model per operator (CPU).
    print(f"{'operator':<22}{'overhead (us)':>15}{'throughput (Melem/s)':>24}{'R^2':>8}")
    for op in sorted(df["op"].unique()):
        m = fit_cost_model(df, op, "cpu")
        print(f"{op:<22}{m.fixed_overhead * 1e6:>15.1f}"
              f"{m.throughput / 1e6:>24.1f}{m.r2:>8.3f}")
    if args.plot:
        from .plots import plot_cost_model

        print(f"\nfigure saved: {plot_cost_model(df, args.plot)}")


def cmd_schedule(args: argparse.Namespace) -> None:
    cpu_df = benchmark(repeats=args.repeats)
    if gpu_available():
        df = cpu_df
        note = "(using MEASURED GPU timings)"
    else:
        # No GPU here: synthesise a GPU cost table to exercise the policy.
        gpu_df = model_gpu_timings(cpu_df)
        import pandas as pd

        df = pd.concat([cpu_df, gpu_df], ignore_index=True)
        note = "(no GPU present -> SYNTHETIC GPU model; see scheduler.model_gpu_timings)"

    sched = SmartScheduler().fit(df)
    print(f"trained ML scheduler on devices {sched._devices} {note}\n")
    print(f"{'operator':<22}{'CPU->GPU crossover (elements)':>32}")
    for op in sorted(df["op"].unique()):
        x = sched.crossover(op)
        label = f"{x:,}" if x else "never (CPU always wins)"
        print(f"{op:<22}{label:>32}")

    # Example placements at three sizes.
    print("\nexample placements:")
    for op in ["spatial_mean", "anomaly"]:
        for n in [50_000, 5_000_000, 60_000_000]:
            p = sched.choose(op, n)
            print(f"  {op:<20} n={n:>12,}  -> {p.device.value:<3}  "
                  f"(predicted {p.speedup:.1f}x vs slowest device)")


def cmd_forecast(args: argparse.Namespace) -> None:
    if not Path(args.path).exists():
        write_netcdf(args.path, n_time=max(args.n_time, 240), chunk_time=1)
    # Derive the index by streaming the file through the spatial-mean reducer.
    proc = InTransitProcessor(spatial_mean_reducer, device=Device.CPU)
    series = proc.run(stream_time_slices(args.path, var="tas")).series

    report = evaluate_forecast(series, horizon=args.horizon, period=12)
    print(f"forecasting global-mean temperature index, horizon={report.horizon} months\n")
    print(f"  model    MAE={report.model_mae:.3f}  RMSE={report.model_rmse:.3f}")
    print(f"  baseline MAE={report.baseline_mae:.3f}  RMSE={report.baseline_rmse:.3f}  "
          f"(seasonal-naive)")
    verdict = "beats" if report.skill_score > 0 else "does not beat"
    print(f"  skill score : {report.skill_score:+.1%}  "
          f"(model {verdict} the baseline)")
    if args.plot:
        from .plots import plot_forecast

        train = series[: -report.horizon]  # what the models actually saw
        print(f"\nfigure saved: {plot_forecast(report, args.plot, history=train)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cwa", description=__doc__)
    p.add_argument("--version", action="version", version=f"cwa {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="write a synthetic CF-NetCDF dataset")
    g.add_argument("--path", default=str(_default_path()))
    g.add_argument("--n-time", type=int, default=240)
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("stream", help="in-transit reduce to a climate index")
    s.add_argument("--path", default=str(_default_path()))
    s.add_argument("--n-time", type=int, default=240)
    s.add_argument("--gpu", action="store_true", help="request GPU (falls back to CPU)")
    s.set_defaults(func=cmd_stream)

    b = sub.add_parser("benchmark", help="time operators + fit performance model")
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--plot", nargs="?", const="cwa_benchmark.png", default=None,
                   metavar="PNG", help="save the cost-model figure (needs [plots] extra)")
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("schedule", help="train ML placement policy, show crossovers")
    c.add_argument("--repeats", type=int, default=3)
    c.set_defaults(func=cmd_schedule)

    f = sub.add_parser("forecast", help="forecast the index vs a baseline")
    f.add_argument("--path", default=str(_default_path()))
    f.add_argument("--n-time", type=int, default=240)
    f.add_argument("--horizon", type=int, default=12)
    f.add_argument("--plot", nargs="?", const="cwa_forecast.png", default=None,
                   metavar="PNG", help="save the forecast figure (needs [plots] extra)")
    f.set_defaults(func=cmd_forecast)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
