# Original endpoint-restricted atlas

This completed run used `interior_search=auto`: Padé3 considered endpoints
unless a quadratic safeguard supplied an interior step. It was the initial
atlas configuration, carried over from RLVR, and is retained for transparency.

The first corrected atlas is `../coarse_reference/`, with `interior_search=true` applied to all
12 tasks and the same seeds, learning-rate grid, epsilon, beta, and budgets.
The current `../reference/` additionally refines flat baselines to 13 rates.
This archive is not evidence about Padé3 with rational interior candidates.

`manifest.json` and all original CSVs are preserved. `sources/` holds copies of
the exact files listed in the original manifest's SHA-256 map.
