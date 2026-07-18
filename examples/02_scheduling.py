"""Methods -- ML-based scheduling across heterogeneous hardware.

Benchmarks the operators, trains the learned placement policy, and shows the
size-dependent device choice plus each operator's learned CPU->GPU crossover.
Also cross-checks the learned crossover against the analytical performance model
for one operator -- two independent methods that should broadly agree.

On a GPU node this uses measured GPU timings automatically. On a CPU-only host
it synthesises a documented GPU cost model (clearly labelled) so the scheduling
logic can be exercised end-to-end.

Run:  python examples/02_scheduling.py
"""

import pandas as pd

from cwa.backend import gpu_available
from cwa.perfmodel import benchmark, crossover_elements, fit_cost_model
from cwa.scheduler import SmartScheduler, model_gpu_timings

if __name__ == "__main__":
    cpu_df = benchmark(repeats=3)

    if gpu_available():
        df, source = cpu_df, "measured GPU timings"
    else:
        df = pd.concat([cpu_df, model_gpu_timings(cpu_df)], ignore_index=True)
        source = "SYNTHETIC GPU model (no GPU on this host)"

    sched = SmartScheduler().fit(df)
    print(f"trained scheduler on devices {sched._devices}  [{source}]\n")

    for op in sorted(df["op"].unique()):
        x = sched.crossover(op)
        label = f"{x:,} elements" if x else "never (CPU wins at every size)"
        print(f"  {op:<22} CPU->GPU crossover: {label}")

    print("\nplacement decisions:")
    for n in [10_000, 200_000, 5_000_000, 60_000_000]:
        p = sched.choose("spatial_mean", n)
        print(f"  spatial_mean, n={n:>12,}  ->  {p.device.value.upper():<3}"
              f"  (predicted {p.speedup:.1f}x vs the slower device)")

    # Cross-validate against the analytical model (learned vs first-principles).
    if not gpu_available():
        cpu_model = fit_cost_model(df, "spatial_mean", "cpu")
        gpu_model = fit_cost_model(df, "spatial_mean", "gpu")
        analytic = crossover_elements(cpu_model, gpu_model)
        learned = sched.crossover("spatial_mean")
        print(f"\n  spatial_mean crossover  analytical={analytic:,.0f}  "
              f"learned={learned:,}  (independent methods, same order of magnitude)")
