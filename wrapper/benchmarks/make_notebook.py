"""Build and execute a notebook that reads a completed evidence directory."""
import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import json
import sys
import tempfile

from .tasks import DESCRIPTIONS


def build(results="benchmarks/results/reference", output="notebooks/benchmark_atlas.ipynb"):
    root = Path(__file__).resolve().parents[1]
    results_path = (root / results).resolve()
    relative = str(results_path.relative_to(root))
    manifest = json.loads((results_path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Wait for the full experiment before generating an executed notebook")
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md("""# ZFO benchmark atlas

**One direction. Four local models. Twelve tests of when step selection is useful.**

This notebook studies SGD, Adam, and Muon with constant learning rates and four
ZFO selectors. It is an exploratory CPU experiment using the reusable library,
not a claim of state-of-the-art performance or a reproduction of the LLM paper.

Each experiment follows the same sequence: a short explanation, a result table,
then learning-rate scans and convergence plots. Tables retain failures.
Full protocol: [benchmarks/README.md](../benchmarks/README.md).

The saved outputs are real executed runs. To recompute optimization, run
`python -m benchmarks.suite --output benchmarks/results/new_run --seeds 3`
from the `wrapper/` directory and update `RESULTS` below. Re-executing this notebook
only regenerates analysis from the specified saved evidence."""),
             code(f"""from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'pyproject.toml').exists() and (p / 'benchmarks').exists())
sys.path.insert(0, str(ROOT))
from IPython.display import display, Markdown, Image
from io import BytesIO
import matplotlib.pyplot as plt
from benchmarks.report import load_results, task_tables, task_figure, comparison_table
RESULTS = ROOT / {relative!r}
data = load_results(RESULTS)
manifest = data['manifest']
display({{k: manifest[k] for k in ['status', 'steps', 'tuning_seeds', 'report_seeds', 'lr_grids', 'beta', 'epsilon', 'interior_search', 'torch', 'dtype', 'tuning_runs', 'report_runs', 'elapsed_seconds']}})"""),
             md("""## Read the comparisons fairly

Rates are selected separately for every optimizer/selector using mean final
validation loss on tuning seeds, then frozen before the reporting seeds. Test
examples are untouched during tuning and step selection. Classical problems
use separate seed-defined instances instead of held-out datasets.

All methods get the same update count. **Flat baselines get 13 learning rates;
each ZFO method gets seven.** Flat adds a geometric midpoint between each pair
of the original rates, preserving the same endpoints (1e-5 and 10). Beta stays
fixed at 3. This gives flat 1.86× as many candidates; it is not an equal tuning
compute comparison. ZFO usually uses three forwards and one backward per
update; flat uses one of each. The
**equal-work** table stops at the same `F + 2B` budget using these selected rates.
This is an explicit cost proxy, not measured FLOPs, and these rates were not
retuned for the shorter compute horizon. Timing excludes monitoring, but includes
the adapter's state-snapshot overhead. Three seeds and one CPU run are modest evidence.
Runs that diverge at or before the work budget are incomplete in the equal-work
table; their last finite checkpoint is not treated as a completed budget result.
Later failures can still have a valid earlier equal-work checkpoint.

Thin plot lines show individual seeds; bold lines show means. Table spreads are
sample standard deviations, not confidence intervals. Crosses in scans mean at
least one tuning seed diverged. An edge rate needs a wider scan before
interpreting superiority. Mean curves require all reporting seeds at a checkpoint.
Individual trajectories end at their own last finite checkpoint. Values below
1e-12 are floored only in log plots.

Muon uses native 2D tensors; classical vectors are represented as (1,d) matrices,
so those cases have degenerate matrix geometry. Network bias terms are augmented
columns. This is distinct from the original LLM Muon/AdamW hybrid.""")]
    policy = manifest["interior_search"]
    candidate_note = (
        "Padé3 compares endpoints and valid interior stationary points of its fitted rational model. "
        "The existing pole and degeneracy safeguards still apply. This differs from the RLVR auto convention."
        if policy == "true" else
        "Padé3's rational model is restricted to endpoints. Under auto, a quadratic fallback can still select an interior point."
    )
    cells.insert(2, md(f"""## Which steps can Padé3 select?

This run explicitly uses **interior_search={policy}**. {candidate_note}
Selecting the best fitted-model candidate does not guarantee the best actual
loss along the interval. The diagnostic column **Model interiors (%)** counts
interior selections made without fallback.

The original atlas used auto by mistake; its saved results remain under
`benchmarks/results/endpoint_reference`. The first interior-enabled run with
seven rates for every method is preserved in `benchmarks/results/coarse_reference`.
The current reference keeps that candidate policy and refines only the flat
baseline LR grid to 13 rates. Seeds, beta, epsilon, and budgets are unchanged.
Interior search is applied to every task rather than selected based on
favorable test results."""))
    for task in manifest["tasks"]:
        title, category, description = DESCRIPTIONS[task]
        cells.extend([
            md(f"## {title}\n\n**{category.title()} experiment.** {description}\n\nAll 15 pairings below use the same seed-defined task instances. Lower loss is better; Δ loss is ZFO minus the paired flat baseline. The equal-work and final columns may favor different choices."),
            code(f"outcomes, diagnostics = task_tables(data, {task!r})\ndisplay(outcomes)\ndisplay(diagnostics.round(4))"),
            code(f"fig = task_figure(data, {task!r})\nbuffer = BytesIO()\nfig.savefig(buffer, format='png', dpi=120)\ndisplay(Image(data=buffer.getvalue()))\nplt.close(fig)"),
        ])
    cells.extend([
        md("""## Where does each variant help, tie, or hurt?

Every ZFO/base pairing remains in this table. Counts compare the reporting seeds
pairwise against the corresponding flat optimizer. A tie is within 1% of flat
loss plus 1e-8. Nonfinite pairs are separate. These are descriptive counts, not
significance tests; no best variant is selected on these reporting results."""),
        code("comparisons = comparison_table(data)\nwith __import__('pandas').option_context('display.max_rows', 160):\n    display(comparisons)"),
        md("""## Reproduction and next experiments

The saved manifest identifies the exact source hashes, software environment,
budgets, and seeds. Preserve it with `scans.csv`, `selected.csv`, `runs.csv`, and
`curves.csv`. A complete rerun never silently overwrites an earlier directory.

Useful follow-ups are five-seed reporting, longer horizons, refined/wider LR
grids for boundary selections, separately tuned compute-matched budgets, and
predeclared epsilon/beta/interior-search ablations. These extensions are not
presented as already completed results. Probe cancellation in float32 and the
memory overhead of state snapshots also need workload-specific evaluation
before moving this reference adapter to large-scale training.""")])
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    output_path = root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Register a temporary kernelspec pointing at the interpreter running this
    # script, never whichever unrelated system kernel happens to be installed.
    with tempfile.TemporaryDirectory() as tmp:
        kernel = Path(tmp) / "zfo-local"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "ZFO local", "language": "python"}))
        from jupyter_client import KernelManager
        km = KernelManager(kernel_name="zfo-local", kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
        NotebookClient(notebook, km=km, timeout=180, resources={"metadata": {"path": str(root)}}).execute()
    nbf.validate(notebook)
    nbf.write(notebook, output_path)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="benchmarks/results/reference")
    parser.add_argument("--output", default="notebooks/benchmark_atlas.ipynb")
    args = parser.parse_args()
    build(args.results, args.output)
