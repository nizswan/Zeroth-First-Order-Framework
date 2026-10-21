# ZFO benchmark atlas

Twelve small problems test where step selection helps and where it costs more
than it gains. The notebook repeats: **explanation → results table → plots**.
The suite includes the six earlier classical problem families (linear and
nonlinear least squares, logistic regression, matrix factorization, Rosenbrock,
Beale), plus an isotropic control, Rastrigin, noisy minibatch regression,
two-moons classification, ReLU XOR, and label corruption. These are fresh
experiments with the reusable wrapper, not a reproduction of the old notebook's
numbers or the paper's LLM tables.

See [FINDINGS.md](FINDINGS.md) for a short reading of the completed reference run,
including favorable cases, regressions, and a result that reverses after counting
probe work.

```bash
python -m pip install -e '.[bench,test]'
python -m benchmarks.suite --output benchmarks/results/my_run --steps 100 --seeds 3
python -m benchmarks.report benchmarks/results/my_run --site ../website
python -m benchmarks.make_notebook --results benchmarks/results/my_run
```

Run these commands from `wrapper/`.

The suite refuses to overwrite an existing manifest. Use a fresh directory for
each run. `--seeds 5` expands both tuning and reporting cohorts to five. For a
quick execution check, use `--tasks sphere noisy_labels --steps 8`; short smoke
runs are not evidence about convergence.

## Fairness contract

1. Compare **SGD, Adam, and native PyTorch Muon**, each with flat LR or one of
   Taylor2, Taylor3, Padé2, Padé3: 15 pairings per task. Tune each pairing
   independently. Each ZFO method gets seven rates: 1e-5, 1e-4, 1e-3, 0.01,
   0.1, 1, 10. Flat baselines get **13 rates**: those seven plus the geometric
   midpoint of every adjacent pair (approximately 3.162e-5, 3.162e-4, 0.003162,
   0.03162, 0.3162, 3.162). This gives flat baselines 1.86× as many candidates
   over the same range. No baseline inherits the selected ZFO learning rate,
   and Muon gets its own selection. This is an intentionally denser baseline
   search, not equal candidate counts or equal tuning compute.
2. Use tuning seeds 100–102 and untouched reporting seeds 200–202 by default.
   Each seed identifies a problem/data realization, initialization, and batch
   sequence; it is paired across methods. Thus SD includes variation between
   synthetic problem realizations, not only initialization on one fixed dataset.
3. Select LR by mean **final validation loss** across all tuning seeds. Any
   failed seed makes a rate ineligible; a tie selects the lower rate. If every
   rate fails, mark the pairing invalid. Never select rates on reporting seeds
   or held-out test data. Classical objectives have no dataset split: tuning
   uses final objective on tuning instances; reporting uses separate instances.
4. Supervised problems have separate train/validation/test examples (192/96/192).
   Probe only the current training batch; keep that batch identical for gradient
   and all probes. Validation/test values never guide the per-step selector.
5. Fix beta=3, epsilon=1e-3, interior_search=true before the sweep. These knobs
   are **not tuned**. Constant base rates are used throughout. All optimizers
   have zero weight decay. SGD has no momentum; Adam uses (0.9, 0.999); Muon uses
   momentum 0.95, Nesterov, five Newton–Schulz steps, match_rms_adamw adjustment.
   Padé3 compares endpoints with valid stationary points of the fitted rational
   model, with the existing pole/degeneracy safeguards. Taylor2/Taylor3 already
   include interior points under auto, so their candidate policy is unchanged.
   This explicit atlas setting differs from the RLVR `auto` endpoint convention.
6. Run 100 updates per trial. Report mean ± **sample SD**, failure counts,
   selected rates, grid-edge warnings, zero steps, fallbacks, training forwards,
   backwards, and measured CPU training time. Failed reporting seeds are not
   dropped from means: a table cell becomes `failed / incomplete`.
7. Plot loss against updates and measured **F + 2B** as an explicit work proxy;
   it is not measured FLOPs. An additional table compares test loss at the last
   checkpoint within 300 work units (100 flat updates, usually 60 ZFO updates).
   These are the same rates selected for the 100-update budget, not a separate
   equal-work tuning sweep. This supplementary view may favor a different
   configuration if retuned at that budget.
   A run that diverges at or before that budget is marked incomplete in the
   equal-work table. Raw CSVs retain its last finite checkpoint for inspection;
   that checkpoint is not counted as a completed budget result. Failures after
   the budget can still have a valid earlier equal-work checkpoint.
8. Monitoring and held-out evaluation forwards are counted separately and
   excluded from the training work/time axes. Total suite wall time is saved.
   Timing includes ZFO state snapshots and is descriptive on one CPU execution,
   not a controlled hardware performance claim. Methods run in a fixed order.
9. Each tensor is genuinely 2D for native Muon; vector tasks use a (1,d) matrix
   and therefore provide only a **degenerate matrix-geometry comparison**.
   Network biases use augmented input columns. Muon is applied to all these
   matrices, including output matrices; this is not the hybrid LLM Muon recipe.
10. Publish every predeclared task and every pairing, including unstable scans.
    An edge-selected LR signals an unresolved search-range limit. Three seeds
    are exploratory evidence, not a significance test or a universal ranking.

## Reading the notebook

The left plot shows validation-rate scans (crosses mark candidates with at
least one divergent tuning seed). Middle/right show reporting-seed training
loss curves at the selected rate. Thin lines show individual seeds, bold lines
show means, and tables report sample SD. Mean curves stop when a checkpoint
lacks any reporting seed; no mean of survivors is substituted. Individual
trajectories remain visible up to their own last finite checkpoint.
The work axis uses the mean measured work at each shared update index because
zero directions can eliminate probes. Log plots floor values at 1e-12 for
display only; raw CSVs retain exact values.

The paired comparison table retains all four ZFO variants. Better/tied/worse
uses a descriptive tolerance of 1% of the paired flat loss plus 1e-8; it is not
a statistical test. There is no selected global winner. A promising result
should be followed by longer budgets, broader/finer scans, five reporting seeds,
and a separate compute-budget tuning study before becoming a paper claim.

`manifest.json` records exact source hashes, Git base revision, environment,
seeds, and settings. `scans.csv` retains all tuning runs; `selected.csv` records
the selection; `runs.csv` contains reporting outcomes; `curves.csv` retains
per-update trajectories. Nonfinite failure outcomes use CSV infinity, not a
made-up large finite score. The site and notebook read these saved results.

## Padé3 candidate-policy correction

The initial atlas accidentally used `interior_search=auto`, importing the RLVR
endpoint restriction into Padé3. It is preserved in
[`results/endpoint_reference/`](results/endpoint_reference/manifest.json), with
snapshots of its hashed source files. Its interior Padé3 steps could only come
from quadratic fallbacks. It should not be used to assess unrestricted Padé3.

The first corrected run is preserved in
[`results/coarse_reference/`](results/coarse_reference/manifest.json), with
`interior_search=true` and the same tasks, seeds, seven-rate grids, epsilon,
beta, and budgets as the endpoint run. This corrects the intended candidate
policy; it does not choose between the two policies using test scores.
`Model interiors (%)` counts selections strictly inside (0,beta) made by the
specified model without a numerical fallback. The selected point is the best
candidate of the fitted local model, not a verified minimum of the actual loss.

## Denser flat-baseline sweep

The current `reference` refines only flat baselines from seven to 13 rates.
The extra candidates are fixed geometric midpoints, not selected from reporting
outcomes. Every ZFO rate grid and beta=3 remain unchanged, as do all seeds,
100-update budgets, optimizer settings, epsilon, and interior-search policy.
There are 4,428 tuning trials and 540 reporting trials. The manifest stores
the exact grid per optimizer and method, and `sources/` preserves its hashed
source files. The earlier seven-rate evidence remains in `coarse_reference`.
