"""Tables and plots from saved runs; never selects a rate using test results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .suite import BASES, METHODS
from .tasks import DESCRIPTIONS

COLORS = dict(zip(METHODS, ("#626b72", "#087f8c", "#c26328", "#6754a2", "#b44166")))
LABELS = {"flat": "Flat LR", "taylor2": "Taylor2", "taylor3": "Taylor3", "pade2": "Padé2", "pade3": "Padé3"}


def load_results(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("This benchmark has not completed; do not present partial runs as a result")
    frames = {name: pd.read_csv(path / f"{name}.csv") for name in ("scans", "runs", "curves", "selected")}
    if len(frames["runs"]) != manifest["report_runs"] or len(frames["scans"]) != manifest["tuning_runs"]:
        raise ValueError("Run counts do not match the completed manifest")
    return {"manifest": manifest, **frames}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return "failed / incomplete"
    return f"{values.mean():.4g} ± {values.std(ddof=1):.2g}"


def equal_work_losses(runs, manifest):
    """Exclude early divergence, while retaining checkpoints before later failure.

    Raw CSVs preserve the last finite checkpoint even when a trial diverges
    before reaching the work budget. That checkpoint is not a budget result.
    """
    completed_budget = (runs.status == "ok") | (runs.work > 3 * manifest["steps"])
    return runs.equal_work_test.where(completed_budget)


def task_tables(data, task):
    rows, diagnostics = [], []
    selected = data["selected"]
    for base in BASES:
        flat = data["runs"].query("task == @task and base == @base and method == 'flat'").set_index("seed")
        for method in METHODS:
            run = data["runs"].query("task == @task and base == @base and method == @method").sort_values("seed")
            pick = selected.query("task == @task and base == @base and method == @method").iloc[0]
            work_losses = equal_work_losses(run, data["manifest"])
            row = {"Base": base, "Selector": LABELS[method], "LR": pick.lr,
                   "Final loss ± SD": mean_std(run.test), "Equal-work loss ± SD": mean_std(work_losses),
                   "Equal-work failed seeds": int((~np.isfinite(work_losses)).sum()),
                   "Failed seeds": int((run.status != "ok").sum()), "Grid edge": bool(pick.grid_edge)}
            if run.test_accuracy.notna().any():
                row["Test accuracy ± SD"] = mean_std(run.test_accuracy)
            rows.append(row)
            paired = run.set_index("seed").test - flat.test
            diagnostics.append({"Base": base, "Selector": LABELS[method],
                                "Δ loss vs flat ± SD": mean_std(paired),
                                "Training seconds ± SD": mean_std(run.train_seconds),
                                "Forward calls (mean)": run.forwards.mean(),
                                "Backward calls (mean)": run.backwards.mean(),
                                "Zero steps (%)": 100 * run.zero_fraction.mean(),
                                "Fallbacks (%)": 100 * run.fallback_fraction.mean(),
                                "Model interiors (%)": 100 * run.model_interior_fraction.mean() if "model_interior_fraction" in run else np.nan,
                                "Tuning valid": bool(pick.tuning_valid)})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def task_figure(data, task):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout="constrained")
    seeds = data["manifest"]["report_seeds"]
    for r, base in enumerate(BASES):
        scan = data["scans"].query("task == @task and base == @base")
        curves = data["curves"].query("task == @task and base == @base")
        for method in METHODS:
            color = COLORS[method]
            subset = scan.query("method == @method")
            rates, means, spreads = [], [], []
            for rate, group in subset.groupby("lr", sort=True):
                rates.append(rate)
                valid = (group.status == "ok").all()
                means.append(group.validation.mean() if valid else np.nan)
                spreads.append(group.validation.std() if valid else np.nan)
            rates, means, spreads = map(np.asarray, (rates, means, spreads))
            style = "--" if method == "flat" else "-"
            axes[r, 0].plot(rates, np.maximum(means, 1e-12), marker="o", ms=3, color=color, ls=style, label=LABELS[method])
            for _, seed_scan in subset.groupby("seed"):
                seed_scan = seed_scan.sort_values("lr")
                values = seed_scan.validation.where(seed_scan.status == "ok").to_numpy()
                axes[r, 0].plot(seed_scan.lr, np.maximum(values, 1e-12), color=color, alpha=0.16, lw=0.8)
            # Mark every failed candidate at the top in axis coordinates.
            failed = ~np.isfinite(means)
            axes[r, 0].plot(rates[failed], np.full(failed.sum(), 0.97 - METHODS.index(method) * 0.035),
                            "x", color=color, ms=4, transform=axes[r, 0].get_xaxis_transform())
            history = curves.query("method == @method")
            # All reporting seeds are required at a checkpoint. No survivor-only means.
            for c, coordinate in ((1, "update"), (2, "work")):
                # The work count can vary when a direction becomes exactly zero.
                # Plot means at shared update indices, with mean measured work on x.
                stats = history.groupby("update").agg(mean=("train_loss", "mean"), sd=("train_loss", "std"),
                                                       n=("seed", "count"), work=("work", "mean"))
                stats.loc[stats.n != len(seeds), ["mean", "sd"]] = np.nan
                x = stats.index.to_numpy() if coordinate == "update" else stats.work.to_numpy()
                y, sd = stats["mean"].to_numpy(), stats.sd.to_numpy()
                axes[r, c].plot(x, np.maximum(y, 1e-12), color=color, ls=style, label=LABELS[method])
                for _, seed_curve in history.groupby("seed"):
                    seed_curve = seed_curve.sort_values("update")
                    axes[r, c].plot(seed_curve[coordinate], np.maximum(seed_curve.train_loss, 1e-12),
                                    color=color, ls=style, alpha=0.18, lw=0.8)
        for c in range(3):
            axes[r, c].set_yscale("log")
            axes[r, c].grid(alpha=0.15)
            axes[r, c].set_ylabel(f"{base} · {'validation' if c == 0 else 'training'} loss")
        axes[r, 0].set_xscale("log")
        axes[r, 0].set_xlabel("Base learning rate · tuning seeds")
        axes[r, 1].set_xlabel("Optimizer updates · reporting seeds")
        axes[r, 2].set_xlabel("Training work proxy F + 2B · reporting seeds")
    axes[0, 0].set_title("Independent learning-rate scans")
    axes[0, 1].set_title("Progress per update")
    axes[0, 2].set_title("Progress per evaluation budget")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    policy = data["manifest"]["interior_search"]
    grids = data["manifest"]["lr_grids"][BASES[0]]
    flat_count = len(grids["flat"]) if isinstance(grids, dict) else len(grids)
    zfo_count = len(grids["taylor2"]) if isinstance(grids, dict) else len(grids)
    fig.suptitle(DESCRIPTIONS[task][0] + "  |  means + individual seed trajectories", fontsize=17)
    fig.supxlabel(f"LR candidates: {flat_count} flat / {zfo_count} ZFO · β={data['manifest']['beta']:g} · Interior search: {policy} · Padé3 stationary points {'enabled' if policy == 'true' else 'disabled'}", fontsize=9)
    return fig


def comparison_table(data):
    """Every pairing retained; descriptive differences, no post-hoc winner claim."""
    records = []
    for task in data["manifest"]["tasks"]:
        for base in BASES:
            flat = data["runs"].query("task == @task and base == @base and method == 'flat'").set_index("seed")
            for method in METHODS[1:]:
                zfo = data["runs"].query("task == @task and base == @base and method == @method").set_index("seed")
                diff = zfo.test - flat.test
                tol = 1e-8 + 0.01 * flat.test.abs()
                finite = np.isfinite(zfo.test) & np.isfinite(flat.test)
                records.append({"Task": DESCRIPTIONS[task][0], "Base": base, "ZFO": LABELS[method],
                                "Better / tied / worse": f"{int(((diff < -tol) & finite).sum())} / {int(((diff.abs() <= tol) & finite).sum())} / {int(((diff > tol) & finite).sum())}",
                                "Nonfinite pairs": int((~finite).sum()), "Paired Δ loss ± SD": mean_std(diff)})
    return pd.DataFrame(records)


def export(path, site_path):
    data = load_results(path)
    site_path = Path(site_path)
    (site_path / "assets" / "plots").mkdir(parents=True, exist_ok=True)
    records = []
    for task in data["manifest"]["tasks"]:
        fig = task_figure(data, task)
        fig.savefig(site_path / "assets" / "plots" / f"{task}.png", dpi=120)
        plt.close(fig)
        table, _ = task_tables(data, task)
        table.to_csv(Path(path) / f"{task}_summary.csv", index=False)
        for row in table.to_dict("records"):
            records.append({"task": task, "name": DESCRIPTIONS[task][0], "category": DESCRIPTIONS[task][1], **row})
    tasks = [{"id": task, "title": DESCRIPTIONS[task][0], "category": DESCRIPTIONS[task][1],
              "description": DESCRIPTIONS[task][2]} for task in data["manifest"]["tasks"]]
    (site_path / "assets" / "results.json").write_text(json.dumps({"manifest": data["manifest"], "tasks": tasks, "rows": records}, indent=2, allow_nan=False) + "\n")
    comparison_table(data).to_csv(Path(path) / "paired_comparisons.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--site", default="../website")
    args = parser.parse_args()
    export(args.results, args.site)
