# What the reference run suggests

These are exploratory observations from the complete saved
[reference evidence](results/reference/manifest.json), using 100 updates,
three tuning seeds, and three distinct reporting seeds. Each optimizer/selector
has its own independently selected rate. Flat baselines receive 13 candidates;
each ZFO variant retains seven, with beta=3, epsilon=1e-3, and interior search
enabled. Numbers below are reporting means ± sample SD. The notebook retains
all 180 task/base/selector configurations, including incomplete outcomes.

## What the denser flat sweep changed

The 13-rate grid preserves every original rate and adds six geometric midpoints
between 1e-5 and 10. It gives flat baselines 1.86× as many candidates as each ZFO
variant. Rates are still selected only on the tuning seeds; optimizer settings,
seeds, and update budgets are unchanged. All ZFO numerical trajectories,
selected rates, and final outcomes match the earlier run exactly, apart from
timing. Earlier evidence and its source snapshots are retained in
[coarse_reference](results/coarse_reference/manifest.json).

The selected flat rate changed in **18 of 36** problem/base combinations. All
18 improved the tuning objective. On reporting seeds, 13 had a lower mean loss,
three had a higher mean, and two became incomplete because a seed diverged.
This is why validation improvement is not a guarantee of reporting improvement.
The new rates are retained even when reporting outcomes are worse; there is no
selection between the old and new rates using reporting seeds. The full
[before/after table](results/reference/flat_refinement.csv) includes every flat
baseline, with validation and reporting losses.

Against these more densely tuned flat baselines, the current comparisons are:

| ZFO variant | Lower mean final loss | Similar | Higher | Incomplete comparison |
| --- | ---: | ---: | ---: | ---: |
| Taylor2 | 15 | 12 | 7 | 2 |
| Taylor3 | 15 | 11 | 8 | 2 |
| Padé2 | 7 | 9 | 17 | 3 |
| Padé3 | 13 | 10 | 11 | 2 |

Each row covers all 36 problem/base combinations. Similar means within 1% of
the flat mean plus 1e-8. These are descriptive comparisons of three-seed means,
not significance tests. An incomplete comparison means either method has at
least one failed reporting seed; it does not necessarily mean ZFO failed.

At the common work budget, the corresponding counts are:

| ZFO variant | Lower mean loss | Similar | Higher | Incomplete comparison |
| --- | ---: | ---: | ---: | ---: |
| Taylor2 | 11 | 7 | 16 | 2 |
| Taylor3 | 11 | 9 | 14 | 2 |
| Padé2 | 6 | 6 | 21 | 3 |
| Padé3 | 8 | 6 | 20 | 2 |

The equal-work checkpoint uses the rates selected for the 100-update budget,
with a common training proxy of 300 = forwards + 2 × backwards. Flat reaches
100 updates; a usual ZFO run reaches 60. This is not measured FLOPs and is not
a separate compute-budget tuning experiment. Monitoring is excluded. A run
that diverges at or before the budget is incomplete; its last finite checkpoint
is retained in the raw CSV but is not a completed budget result. This reporting
correction applies equally to flat and ZFO runs.

## Padé3 interior search remains enabled

The current reference explicitly uses **interior_search=true**, including Padé3
stationary points. The initial atlas accidentally used the RLVR auto convention,
which restricted the rational model to endpoints. That original run is preserved
in [endpoint_reference](results/endpoint_reference/manifest.json); its Padé3
results should not be described as unrestricted interior search.

The isolated policy correction, before refining the flat LR grid, gave:

| Historical Padé3 policy, seven rates for every method | Lower mean final loss | Similar | Higher | Incomplete comparison |
| --- | ---: | ---: | ---: | ---: |
| Original auto (rational endpoints) | 15 | 5 | 15 | 1 |
| Corrected true (rational interiors enabled) | 22 | 6 | 8 | 0 |

These historical counts use the earlier seven-rate flat baselines. Use the
current tables above for comparisons against the 13-rate flat sweep.

Padé3 selected **2,709 rational interior steps out of 10,800 updates**, occurring
in 83 of 108 reporting runs. These exclude quadratic fallbacks and are unchanged
by refining the flat grid. Selecting the fitted model's best candidate still
does not guarantee the best actual objective value along the line.

## Examples and limitations

| Problem / base | Flat, 100 updates | Taylor2, 100 updates | Taylor2, equal-work checkpoint | Interpretation |
| --- | --- | --- | --- | --- |
| Rosenbrock / Adam | 0.7948 ± 0.1354 | 0.0008446 ± 0.000731 | 0.03022 ± 0.02384 | A favorable case in both views in this run |
| Ill-conditioned least squares / Adam | 0.0004878 ± 0.000399 | 0.0002075 ± 0.000228 | 0.002110 ± 0.001239 | Better per update, worse when probe work is counted |
| Beale / Adam | 0.008503 ± 0.002459 | 0.8062 ± 0.05206 | 0.8988 ± 0.09493 | A substantial loss against the tuned flat baseline |
| Noisy minibatch regression / Muon | 0.2530 ± 0.01366 | 0.2858 ± 0.03887 | 0.4058 ± 0.03350 | No improvement in the mean; uncertainty remains large |

Other useful controls and failure cases:

- **Isotropic quadratic:** tuned flat SGD reaches zero loss. Step selection has
  no room to improve its final objective, and its probes add overhead.
- **Rastrigin:** all methods can remain trapped in nonzero local minima. A
  better step along a chosen direction does not solve global optimization.
- **Reporting failures:** flat SGD fails on Rosenbrock seed 202 and Beale seed
  201 at the newly selected rates. SGD + Padé2 retains its failures on matrix
  factorization seed 201 and Beale seed 200. Their means are incomplete, not
  averages over surviving seeds. Padé3 has no reporting failures.
- **Scan limits:** 17 of 180 selected configurations lie at a grid boundary.
  The finer flat sweep keeps the same range, so those range limits remain.
  Across all tuning candidates, 170 of 4,428 trials diverged; these are retained
  in the raw CSV.
- **Small supervised models:** logistic regression, two moons, ReLU XOR, and
  label corruption distinguish training progress from held-out performance.
  Their results depend on the optimizer and selector; they are not evidence
  that ZFO improves every network.

Longer budgets, wider boundary scans, independent tuning at a fixed work
budget, and five reporting seeds remain useful follow-ups. The current atlas
keeps the agreed beta and optimizer parameters fixed; it is not a joint
hyperparameter search or a universal optimizer ranking.
