"""Restyle saved evidence for the website; no training, tuning, or smoothing.

Run from the repository root: python -m benchmarks.site_figures
The earlier six-problem experiment has its own cached data and provenance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
COLORS = ["#59645f", "#2c6685", "#367447", "#b16b28", "#885a93", "#ba4b40"]
METHODS = ["flat", "taylor2", "taylor3", "pade2", "pade3"]
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
                    "ytick.labelsize": 10, "font.family": "DejaVu Sans",
                    "text.color": "#35463e", "axes.labelcolor": "#35463e",
                    "xtick.color": "#59645f", "ytick.color": "#59645f",
                    "axes.edgecolor": "#ccd5c6", "axes.spines.top": False,
                    "axes.spines.right": False, "svg.fonttype": "none",
                    "svg.hashsalt": "zfo-atlas", "figure.facecolor": "#fffef9",
                    "axes.facecolor": "#fffef9", "path.simplify": False})


def canvas():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.subplots_adjust(left=.15, right=.97, top=.94, bottom=.21)
    ax.grid(axis="y", color="#e2e7dc", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    return fig, ax


def save(fig, path):
    fig.savefig(path, metadata={"Date": None})
    # Keep text selectable, with a browser-safe fallback on machines without DejaVu.
    svg = path.read_text().replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif")
    path.write_text(svg)
    plt.close(fig)


def classical():
    folder = ASSETS / "classical"
    saved = np.load(folder / "saved_perturbed_curves.npz")
    slugs = ["linear", "nonlinear", "logistic", "matrix", "rosenbrock", "beale"]
    for index, slug in enumerate(slugs):
        fig, ax = canvas()
        for method, y in zip(saved["methods"], saved["median_log10"][index]):
            color_index = (0 if not method.startswith(("ZFO", "GeN")) else
                           5 if method.startswith("GeN") else
                           {"T2": 1, "T3": 2, "P2": 3, "P3": 4}[method.split("-")[1]])
            ax.plot(saved["iterations"], y, color=COLORS[color_index],
                    ls="--" if method.endswith("SGD") else "-", lw=1.65)
        ax.set(xlim=(0, 100), xlabel="Updates", ylabel="Median log₁₀ objective")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        save(fig, folder / f"{slug}.svg")


def atlas():
    source = ROOT / "benchmarks/results/reference"
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    curves = pd.read_csv(source / "curves.csv")
    runs = pd.read_csv(source / "runs.csv")
    folder = ASSETS / "atlas"
    folder.mkdir(exist_ok=True)
    for task in manifest["tasks"]:
        for base in ["SGD", "Adam", "Muon"]:
            for coordinate in ["update", "work"]:
                fig, ax = canvas()
                failures = []
                for index, method in enumerate(METHODS):
                    history = curves.query("task == @task and base == @base and method == @method")
                    stats = history.groupby("update").agg(mean=("train_loss", "mean"),
                              n=("seed", "count"), work=("work", "mean"))
                    stats.loc[stats.n != len(manifest["report_seeds"]), "mean"] = np.nan
                    style = "--" if method == "flat" else "-"
                    for _, seed in history.groupby("seed"):
                        seed = seed.sort_values("update")
                        ax.plot(seed[coordinate], np.maximum(seed.train_loss, 1e-12),
                                color=COLORS[index], ls=style, lw=.65, alpha=.18)
                    x = stats.index if coordinate == "update" else stats.work
                    ax.plot(x, np.maximum(stats["mean"], 1e-12), color=COLORS[index], ls=style, lw=1.9)
                    failed = runs.query("task == @task and base == @base and method == @method and status != 'ok'")
                    if len(failed):
                        failures.append((index, method, len(failed)))
                for row, (index, method, count) in enumerate(failures):
                    label = ["Flat LR", "Taylor2", "Taylor3", "Padé2", "Padé3"][index]
                    ax.text(.98, .96 - row * .09, f"{label}: {count}/3 failed", transform=ax.transAxes,
                            ha="right", va="top", color=COLORS[index], fontsize=9,
                            bbox={"facecolor": "#fffef9", "edgecolor": "none", "alpha": .9})
                ax.set_yscale("log")
                ax.set_xlabel("Updates" if coordinate == "update" else "Evaluation work · F + 2B")
                ax.set_ylabel("Training loss")
                save(fig, folder / f"{task}-{base.lower()}-{coordinate}.svg")


if __name__ == "__main__":
    classical()
    atlas()
    print("Generated six classical figures and 72 atlas views from saved results.")
