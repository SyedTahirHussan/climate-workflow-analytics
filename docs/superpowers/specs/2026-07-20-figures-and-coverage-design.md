# Figures + coverage pass — design

**Date:** 2026-07-20
**Goal:** Make results visible (publication-style figures from the CLI) and
raise test rigor on the least-covered modules (`forecast.py` 24%,
`perfmodel.py` 30%, `cli.py` 0%).

## A. Figures

- **New module `src/cwa/plots.py`** — plotting is isolated here; the science
  modules keep zero knowledge of matplotlib.
  - `plot_forecast(report, path, history=None)` — training-tail context, then
    held-out truth vs ML forecast vs seasonal-naive baseline, annotated with
    MAE and skill score.
  - `plot_cost_model(df, path)` — log–log measured timings with fitted
    `t(n) = overhead + n/throughput` curves per operator, one panel per device.
- **CLI**: `cwa forecast --plot [PNG]` and `cwa benchmark --plot [PNG]`
  (off by default; bare flag uses a sensible default filename).
- **Dependency**: `matplotlib` as optional extra `[plots]`, folded into `[dev]`
  so CI exercises it. Lazy import inside `plots.py` with an actionable error —
  mirrors the CuPy pattern.
- **README**: generated PNGs committed under `docs/img/` and embedded in the
  results section.

## B. Coverage

Property-style tests in the house style:

- `tests/test_forecast.py` — seasonal-naive exact on a noise-free cycle;
  `build_features` uses only past values (corrupting the future must not
  change earlier rows); chronological split; too-short series raises;
  `skill_score` edge cases.
- `tests/test_perfmodel.py` — `fit_cost_model` recovers known
  overhead/throughput from synthetic timings; negative-intercept clamp;
  `crossover_elements` matches a hand-solved answer / returns `None` when one
  device dominates; `benchmark` produces a tidy table on a tiny grid.
- `tests/test_cli.py` — `gen`/`stream`/`forecast` end-to-end via `main([...])`
  on a small temp dataset; `--plot` writes a real PNG (Agg backend).
- `tests/test_plots.py` — both plot functions write non-empty PNGs from
  synthetic inputs.

## Verification

pytest + ruff green locally; figures regenerate from the CLI; CI green after
push.
