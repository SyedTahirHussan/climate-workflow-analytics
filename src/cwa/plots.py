"""Figures for the pipeline -- forecast skill and the fitted cost model.

Plotting lives in this one module so the science code stays free of
presentation concerns. matplotlib is an optional dependency
(``pip install "climate-workflow-analytics[plots]"``), imported lazily so
``import cwa`` works without it -- the same pattern used for CuPy.

Design choices: one colorblind-validated categorical palette with a fixed
slot-per-entity assignment (never cycled), neutral ink for observed/reference
data, recessive grids, no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .forecast import ForecastReport
from .perfmodel import fit_cost_model

# Categorical palette validated for color-vision-deficiency separation
# (adjacent-pair CVD deltaE >= 8). Slots are assigned to entities in this fixed
# order; observed/reference data wears neutral ink instead of a slot.
_PALETTE = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_INK = "#0b0b0b"  # primary text and observed data
_INK_MUTED = "#52514e"  # secondary text, axes
_GRID = "#e8e7e3"
_SURFACE = "#fcfcfb"


def _plt():
    """Import pyplot lazily with an actionable error when it is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "plotting requires matplotlib -- install with "
            "pip install 'climate-workflow-analytics[plots]'"
        ) from exc
    return plt


def _style(ax) -> None:
    """House style: recessive grid and spines, ink-colored labels."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_MUTED)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_MUTED, labelcolor=_INK)


def plot_forecast(
    report: ForecastReport,
    path: str | Path,
    history: np.ndarray | None = None,
) -> Path:
    """Held-out truth vs ML forecast vs seasonal-naive baseline.

    ``history`` (the training series) adds context: its tail is drawn in muted
    ink to the left of the forecast window, so the reader sees the seasonal
    pattern the models had to extrapolate.
    """
    plt = _plt()
    path = Path(path)
    h = report.horizon

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=160)
    fig.patch.set_facecolor(_SURFACE)
    _style(ax)

    x_fc = np.arange(1, h + 1)
    if history is not None:
        hist = np.asarray(history, dtype=float)
        tail = hist[-2 * h :] if hist.size > 2 * h else hist
        x_hist = np.arange(1 - tail.size, 1)
        ax.plot(x_hist, tail, color=_INK_MUTED, lw=1.5, label="observed (training tail)")
        ax.plot(  # dotted bridge from last training point into the window
            [0, 1], [tail[-1], report.y_true[0]], color=_INK_MUTED, lw=1.0, ls=":"
        )
        ax.axvspan(0.5, h + 0.5, color="#f1f0ec", zorder=0, label="held-out window")

    ax.plot(x_fc, report.y_true, color=_INK, lw=2, marker="o", ms=5, label="held-out truth")
    ax.plot(
        x_fc, report.y_model, color=_PALETTE[0], lw=2, marker="s", ms=5,
        label=f"ML forecast (MAE {report.model_mae:.3f})",
    )
    ax.plot(
        x_fc, report.y_baseline, color=_PALETTE[1], lw=2, ls="--", marker="^", ms=5,
        label=f"seasonal-naive (MAE {report.baseline_mae:.3f})",
    )

    ax.set_xlabel("months relative to end of training")
    ax.set_ylabel("global-mean temperature index (\N{DEGREE SIGN}C)")
    ax.set_title(
        f"{h}-month forecast \N{EM DASH} skill score {report.skill_score:+.0%} "
        "vs seasonal-naive",
        color=_INK, loc="left",
    )
    ax.legend(frameon=False, labelcolor=_INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_cost_model(df: pd.DataFrame, path: str | Path) -> Path:
    """Measured timings (dots) with fitted ``t(n) = a + n/T`` curves (lines).

    One panel per device, log-log axes (the sweep spans orders of magnitude,
    which is exactly what makes the overhead floor and the throughput asymptote
    visible as two straight regimes). Each operator keeps one palette slot
    across panels so identity follows the entity, not the panel.
    """
    plt = _plt()
    path = Path(path)

    devices = sorted(df["device"].unique())
    ops = sorted(df["op"].unique())
    if len(ops) > len(_PALETTE):
        raise ValueError(f"more operators ({len(ops)}) than palette slots")
    colors = dict(zip(ops, _PALETTE, strict=False))

    fig, axes = plt.subplots(
        1, len(devices), figsize=(5.8 * len(devices), 4.5), dpi=160,
        sharey=True, squeeze=False,
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, device in zip(axes[0], devices, strict=True):
        _style(ax)
        for op in ops:
            sub = df[(df["op"] == op) & (df["device"] == device)]
            if sub.empty:
                continue
            model = fit_cost_model(df, op, device)
            n = sub["n_elements"].to_numpy(dtype=float)
            ax.scatter(n, sub["seconds"], s=28, color=colors[op], zorder=3)
            grid_n = np.geomspace(n.min(), n.max(), 128)
            ax.plot(
                grid_n, [model.predict(v) for v in grid_n],
                color=colors[op], lw=2,
                label=f"{op} (R\N{SUPERSCRIPT TWO}={model.r2:.2f})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("problem size (elements)")
        ax.set_title(
            f"{device.upper()} \N{EM DASH} measured (dots), fitted t(n) = a + n/T",
            color=_INK, loc="left", fontsize=10,
        )
    axes[0][0].set_ylabel("wall time (s)")
    axes[0][0].legend(frameon=False, labelcolor=_INK, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
