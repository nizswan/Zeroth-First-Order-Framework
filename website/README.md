# GitHub Pages site

`index.html`, `styles.css`, `site.js`, and `assets/` form a buildless static site.
The page presents the paper, configuration builder, search-interval explanation,
a browser-based function playground, and a full-width invitation to try the library.

Preview from the repository root:

```bash
python -m http.server 8765 --directory website
```

Open `http://localhost:8765`. Relative asset paths also work under the repository
subpath used by GitHub Pages. Use the HTTP preview to check the page and its assets.

The `.github/workflows/pages.yml` workflow publishes `website/` when site changes are
pushed to main (or manually dispatched). In the repository
settings, choose **Pages → Build and deployment → Source: GitHub Actions**.

Benchmark assets are generated from completed saved runs:

```bash
cd wrapper
python -m benchmarks.report benchmarks/results/reference --site ../website
```

The public code/notebook links target the main branch.
The Paper link opens the supplied PDF at `assets/paper.pdf`; arXiv links to
`https://arxiv.org/abs/2610.02190`. TACO uses `from ZFO import TACO`, the matrix-only dense-history adapter
documented in [the library guide](../wrapper/docs/library.md).
The configuration builder uses one displacement-based search interval and exposes
interior search plus the library-only `pade3_21` extension. The interval explanation
separates the learning rate’s role in forming the displacement from the bound’s
role in scaling it. Saved benchmark data and source snapshots are preserved.

The classical figures and benchmark atlas are no longer displayed on the page.
Their saved assets remain available. The classical plots use the preserved earlier experiment in
`assets/classical/saved_perturbed_curves.npz`; `provenance.json` documents its
same-seed tuning and carry-forward policy. These are not the newer atlas results.

Rebuild the website's responsive SVG figures from saved data with:

```bash
python website/scripts/site_figures.py
```

Requires NumPy, pandas and Matplotlib. This writes six classical plots and 72
atlas views (12 tasks × 3 optimizers × update/work axes), without rerunning or
changing experiments. The saved atlas figures show means and individual reporting seeds,
and mark failed runs; full plots and result data remain in the assets directory.

The opening illustration adapts the paper's Figure 1 into a loss-minimization
landscape and its matching one-dimensional slice. Regenerate its desktop and
stacked mobile SVGs with `python website/scripts/generate_hero_figure.py` from the
repository root (standard library only). Both views share the synthetic objective
and a cubic model fitted from the current value, slope, and symmetric probes.
The illustration uses displacement multiplier `a = s / s_base`, so the base
optimizer step is `a = 1`. The purple surface marker shows the actual loss at
the model-selected parameters; the purple 2D marker shows the model prediction.
The illustration is independent of the configuration controls and benchmark data.

## Function playground

`playground-core.js` parses bounded math expressions without evaluating JavaScript.
It supports arithmetic, powers, sin/cos/tan/exp/log/sqrt, x, pi, and e. Automatic
differentiation supplies the first derivative; two symmetric function evaluations
estimate the second and third derivatives using the paper's formulas.

The playground uses the paper's physical step coordinate s, unit direction z,
and radius R = beta * learning_rate * abs(fprime(x)). It compares Taylor2,
Taylor3, Pade2 [1/1], and the methodology's Pade3 [2/1], including stationary
points and Taylor2 fallbacks for invalid rational fits or interval poles. This
Pade3 form corresponds to the library's `pade3_21` option; the configuration
builder also offers the separate `pade3` [1/2] implementation. Minimize and
maximize goals are both available. The graph samples the curves for display;
selection uses analytic stationary points and endpoints, not plot samples.

Run the parser, derivative, fitting, domain, and selection checks with:

```bash
node --test website/tests/test_site_playground.cjs
```
