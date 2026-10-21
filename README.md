# ZFO: Zeroth and First-Order Optimization

<p align="center">
  <a href="https://nizswan.github.io/Zeroth-First-Order-Framework/">
    <img src="https://img.shields.io/badge/Project-Website-173e37?style=for-the-badge" alt="Project Website">
  </a>
  <a href="https://nizswan.github.io/Zeroth-First-Order-Framework/assets/paper.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-a54b29?style=for-the-badge" alt="Paper PDF">
  </a>
  <a href="https://arxiv.org/abs/2610.02190">
    <img src="https://img.shields.io/badge/arXiv-2610.02190-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
</p>

<p align="center">
  <strong>Your optimizer picks the direction. ZFO picks the step.</strong>
</p>

<p align="center">
  If you find ZFO useful, please give the project a ⭐ on GitHub!
</p>

<p align="center">
  <a href="https://github.com/nizswan/Zeroth-First-Order-Framework">
    <img src="https://img.shields.io/badge/Support%20ZFO-Star%20on%20GitHub-173e37?style=for-the-badge&logo=github" alt="Support ZFO — Star on GitHub">
  </a>
</p>

---

## Overview

ZFO separates two decisions in optimization: **which direction to move** and
**how far to go**. A first-order optimizer supplies the direction. Taylor or
Padé models use local information along that direction to choose a step within
a bounded search interval.

**[Explore the project website →](https://nizswan.github.io/Zeroth-First-Order-Framework/)**
Read the visual introduction, configure ZFO, or enter your own function in the
interactive playground to compare Taylor and Padé fits.

## Repository layout

The repository has three components. Choose the one that matches your task:

| Component | Purpose | Start here |
| --- | --- | --- |
| **`wrapper/`** | Installable PyTorch optimizer wrapper for supervised training and other differentiable scalar objectives; includes examples, tests, and benchmark evidence | [Wrapper overview](wrapper/README.md) · [API guide](wrapper/docs/library.md) |
| **`paper/`** | Large-model experiment runner, datasets, baselines, pinned dependencies, and reproduction instructions | [Paper experiment guide](paper/README.md) |
| **`website/`** | Static project website, interactive function playground, paper PDF, and figures | [Live website](https://nizswan.github.io/Zeroth-First-Order-Framework/) · [Website guide](website/README.md) |

```text
Zeroth-First-Order-Framework/
├── wrapper/
│   ├── ZFO/                 # Public API and shared mathematical selectors
│   ├── examples/            # Download-free supervised example
│   ├── docs/library.md      # Detailed wrapper API and limitations
│   ├── benchmarks/          # Benchmark suite, protocol, and saved evidence
│   ├── notebooks/           # Benchmark walkthrough
│   ├── tests/               # Optimizer and benchmark behavior checks
│   └── pyproject.toml       # Installable zeroth-first-order package
├── paper/
│   ├── run.py               # Experiment configuration, training, and evaluation
│   ├── zfo.py               # Paper-specific driver using shared selectors
│   ├── baselines.py         # Optimizers and comparison methods
│   ├── prep.py              # Models, datasets, prompts, and reward functions
│   ├── requirements.txt     # Pinned reproduction dependencies
│   ├── tests/               # Paper-driver checks
│   └── README.md            # Paper-specific reference
├── website/
│   ├── index.html           # Project page
│   ├── playground-core.js   # Function parsing and local fitting
│   ├── site.js              # Configuration builder and playground interface
│   ├── styles.css           # Website styling
│   ├── assets/              # Paper PDF and figures
│   ├── scripts/             # Figure generation
│   └── tests/               # Playground checks
└── .github/workflows/       # Library checks and GitHub Pages deployment
```

The wrapper and paper runner share the selectors in
[`wrapper/ZFO/_search.py`](wrapper/ZFO/_search.py), but their training loops and
objective construction remain distinct. Install the wrapper before executing
the paper experiments. The website runs independently in a browser.

### Contents

- [Reusable PyTorch wrapper](#reusable-pytorch-wrapper)
- [Wrapper benchmarks and saved evidence](#wrapper-benchmarks-and-saved-evidence)
- [Paper experiment package](#paper-experiment-package)
- [Usage and hyperparameters](#usage-and-hyperparameters)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Implementation fidelity and limits](#implementation-fidelity-and-limits)
- [Project website and function playground](#project-website-and-function-playground)
- [Repository checks](#repository-checks)

> **Reproduction scope.** The paper experiment package preserves default
> numerical logic and explicit settings; it does not certify that fresh runs
> will match the published means. The historical environment and selected-run
> limitations are documented below. The reusable wrapper is a separate
> fixed-objective interface and does not guarantee that every chosen step
> improves the actual loss.

---

## Reusable PyTorch wrapper

Use the `ZFO` library to add step selection to your own supervised training or
fine-tuning loop. It wraps an existing PyTorch optimizer while leaving your
model, data loader, and scalar loss function under your control.

The repository provides two separate entry points:

| Use case | Entry point | Dependencies |
| --- | --- | --- |
| Supervised training, fine-tuning, or another differentiable scalar objective | `from ZFO import ZFO` | Install the package with `python -m pip install -e ./wrapper` |
| Large-model experiment reproduction | `python paper/run.py ...` | Install `paper/requirements.txt` and the wrapper; follow the reproduction guide below |

### Install the wrapper

The package requires **Python 3.10 or newer** and **PyTorch 2.9 or newer**.
Its distribution name is `zeroth-first-order`; its import name is `ZFO`.
PyTorch is the only core dependency. No PyPI release is implied.

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/nizswan/Zeroth-First-Order-Framework.git#subdirectory=wrapper"
```

For the examples, tests, and benchmark tools, use a checkout. Run the following
from the directory where you want to keep the repository:

```bash
git clone https://github.com/nizswan/Zeroth-First-Order-Framework.git
cd Zeroth-First-Order-Framework
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./wrapper
```

If you already have a checkout, start at its root and skip the clone. The
wrapper commands in this section run from the **repository root**. On Windows,
use the virtual environment's corresponding activation command.

For GPU training, install the PyTorch build appropriate for your CUDA setup.
The large-model reproduction dependencies in `paper/requirements.txt` are separate
from the reusable wrapper's core requirements.

### A complete supervised example

This small regression example generates its own data and needs no downloads:

```python
import torch
from ZFO import ZFO

# Synthetic supervised regression.
torch.manual_seed(17)
x = torch.randn(256, 6, dtype=torch.float64)
y = x @ torch.randn(6, 1, dtype=torch.float64)
y += 0.1 * torch.randn(256, 1, dtype=torch.float64)

model = torch.nn.Linear(6, 1).double()
base = torch.optim.Adam(model.parameters(), lr=0.03)
optimizer = ZFO(
    base,
    model=model,
    method="taylor2",
    epsilon=1e-3,
    beta=3,
    interior_search="auto",
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=60)

for step in range(60):
    # Return the loss only. ZFO handles zero_grad() and backward().
    loss = optimizer.step(lambda: (model(x) - y).square().mean())
    scheduler.step()

    if step % 15 == 0:
        print(f"step={step:02d} loss_before_step={loss.item():.6f}")

with torch.no_grad():
    print(f"final loss={((model(x) - y) ** 2).mean().item():.6f}")
```

For minibatch regression or classification, take each batch **outside** the
closure and return your differentiable scalar loss inside it:

```python
for inputs, targets in loader:
    loss = optimizer.step(lambda: loss_fn(model(inputs), targets))
    scheduler.step()
```

Attach learning-rate schedulers to the **base optimizer**, then advance them
after the ZFO step. `step()` returns the loss from the initial forward pass,
not a fresh evaluation at the selected parameters.

The complete example is in [`wrapper/examples/supervised.py`](wrapper/examples/supervised.py).
See [`wrapper/docs/library.md`](wrapper/docs/library.md) for the detailed library guide.

### Wrapper options

The library uses Python arguments; these are distinct from the experiment
runner's command-line flags documented later in this README.

| Argument | Default | Meaning |
| --- | --- | --- |
| `optimizer` | Required | An existing closure-free PyTorch optimizer, such as SGD, Adam, AdamW, matrix-only Muon, or the included matrix-only TACO |
| `model` | `None` | Pass your model so ZFO can restore mutable buffers around probes, especially with BatchNorm |
| `method` | `"taylor2"` | `taylor2`, `taylor3`, `pade2` [1/1], `pade3` [1/2], or `pade3_21` [2/1] |
| `epsilon` | `1e-3` | Physical parameter-space probe distance; separate from an optimizer's numerical epsilon |
| `beta` | `1.0` | Maximum multiple of the base optimizer's proposed update; the example explicitly uses `3` |
| `interior_search` | `"auto"` | Whether each model considers stationary points inside the interval, as described below |
| `lr` | `None` | Optional override of every base-optimizer group's learning rate; omit it to preserve group-specific rates |

`interior_search="auto"` uses Taylor stationary points, Padé endpoints, and
possible interior quadratic fallbacks. `True` additionally enables stationary
points for both Padé3 forms after their denominator checks. `False` restricts
all models and fallbacks to endpoints. Padé2 has no isolated stationary point
when its denominator is regular.

`pade3_21` is the additional [2/1] library option. `pade3` continues to mean
[1/2]; do not substitute one for the other when reproducing an experiment.
The adapter retains the historical Padé2 degeneracy and denominator handling.
For the general benchmark atlas, the documented setting is explicitly
`interior_search=True`; the original RLVR runner retains `auto`.

### Closure contract and update behavior

- Return the **same differentiable scalar objective on the same fixed batch**
  on every closure call within a step. Do not advance a data iterator, call
  `backward()`, or call `zero_grad()` inside the closure.
- ZFO obtains one tentative base-optimizer update, evaluates two symmetric
  probes, and selects a multiplier between `0` and `beta`. A multiplier of
  `1` takes the usual update; `0` keeps the parameters in place.
- Base-optimizer momentum and moments advance once, even if the selected
  multiplier is zero. Weight decay is included in the proposed update and
  scales with the selected multiplier. This is not equivalent to simply
  replacing the learning-rate scheduler.
- A usual step costs **three forward passes and one backward pass**. A zero
  displacement needs only the original forward. Probes can lie outside the
  final nonnegative search interval.
- **The selected point is not checked with an extra forward pass.** A local
  model can choose a point with worse actual loss. A numerical fallback is
  not an acceptance test or a guarantee of improvement.
- Inspect `optimizer.last_step` for the selected `multiplier`, effective
  group learning rates, forward counts, and fallback diagnostics.

Unlike the preserved large-model reproduction path described below, the
wrapper requires the gradient and probe values to come from the **same scalar
objective**. It does not reproduce the mixed-objective GRPO construction by
itself.

### Supported patterns and limitations

The wrapper supports single-device CPU, CUDA, or MPS training with
**float32 or float64 parameter storage**, multiple parameter groups, frozen
parameters, native weight decay, and base optimizers with `maximize=True`.
All groups must agree on maximization. The included numerical benchmarks use
CPU float64; validate accelerator behavior on your intended hardware. Float32
may need a larger probe distance to reduce finite-difference cancellation.

Torch RNG states are replayed around probes, and passing `model` lets ZFO
restore its mutable buffers. The final Torch RNG and model buffers reflect one
training forward. Python/NumPy randomness and external side effects are not
replayed: prepare augmentations, sampled rollouts, and other stochastic inputs
outside the closure.

This is a **closure-based adapter, not a `torch.optim.Optimizer` subclass** or
a drop-in replacement for a trainer that supplies precomputed gradients. It
does not implement AMP/GradScaler, distributed or sharded training, compiled
or captured steps, sparse gradients, LBFGS, or gradient accumulation.
Single-device support does not imply support for every training framework.

Muon accepts 2D parameters only. The included `from ZFO import TACO` adapter
is also matrix-only and uses dense float32/float64 history. It does not
reproduce TACO's compressed history, gradient-release hooks, auxiliary AdamW,
or architecture-specific routing. A hybrid multi-optimizer adapter is not
included; consult the [library guide](wrapper/docs/library.md#taco-library-adapter)
for the TACO-specific configuration and reference implementation.

Differentiable fixed-rollout policy objectives can use this interface too:
return the negative reward surrogate for minimization, or use a maximizing
base optimizer. Reward construction, sampling, and KL terms remain the caller's
responsibility; `paper/run.py` remains the separate LLM reproduction path.

### Checkpoints, diagnostics, and checks

Save the model and wrapper states, along with the scheduler, RNG, sampler
position, and any other application state needed to resume training:

```python
checkpoint = {
    "model": model.state_dict(),
    "zfo": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "rng": torch.get_rng_state(),
}

# Recreate the same model and parameter groups before loading.
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["zfo"])
scheduler.load_state_dict(checkpoint["scheduler"])
torch.set_rng_state(checkpoint["rng"])
```

Save device RNG states separately when using an accelerator. If a closure or
probe raises an exception or returns a nonfinite value, the adapter restores
parameters, gradients, optimizer state, model buffers, and Torch RNG. These
per-step snapshots add memory and Python overhead; profile large models before
adopting this reference implementation in production.

To run the supervised example and wrapper behavior checks:

```bash
# From the repository root.
python wrapper/examples/supervised.py
python wrapper/examples/supervised.py --optimizer AdamW --method pade3_21 --interior-search true --beta 3
python -m pip install -e './wrapper[test]'
python -m pytest -q wrapper/tests/test_optimizer.py wrapper/tests/test_taco.py
```

The example also accepts `--lr`, `--epsilon`, and `--interior-search`.
Checkpoint loading validates saved settings and parameter groups; an invalid
checkpoint leaves the current optimizer and wrapper state intact. Checkpoints
requesting the removed alternate `search_type` mode require their original
source, or an explicit migration that may change numerical decisions.

These are validation commands, not a claim that a particular GPU setup or
full-scale fine-tuning workload has been validated. The reproduction package's
separate validation scope and historical limitations remain documented below.

---

## Wrapper benchmarks and saved evidence

The wrapper includes a twelve-problem benchmark atlas covering smooth controls,
curved valleys, matrix problems, noisy regression, classification, and small
neural networks. These are **exploratory CPU experiments with the reusable
wrapper**, not reproductions of the paper's LLM tables or the earlier classical
notebook's numbers.

| Resource | Contents |
| --- | --- |
| [Benchmark notebook](wrapper/notebooks/benchmark_atlas.ipynb) | Problem explanations, learning-rate scans, tables, and plots |
| [Benchmark protocol](wrapper/benchmarks/README.md) | Selection rules, seed cohorts, failure handling, and work accounting |
| [Reference findings](wrapper/benchmarks/FINDINGS.md) | Gains, regressions, failures, and compute tradeoffs |
| [Reference results](wrapper/benchmarks/results/reference/) | Saved runs, selected configurations, manifests, and source snapshots |

The reference comparison covers SGD, Adam, and Muon, each with flat learning
rate or Taylor2, Taylor3, Padé2, and Padé3. Every pairing gets its own scan.
Flat baselines receive 13 learning rates and ZFO variants receive seven, both
spanning `1e-5` to `10`. Tuning seeds `100–102` are separate from reporting seeds
`200–202`. The fixed settings are `beta=3`, `epsilon=1e-3`, and
`interior_search=True`, including interior Padé3 candidates. This differs from
the paper runner's default `auto` policy.

The protocol reports failed runs, sample standard deviations, selected grid
boundaries, and extra forward work. Failed reporting seeds are not dropped to
produce a mean of survivors. The `F + 2B` work axis is a proxy, not measured
FLOPs. Equal-work comparisons reuse learning rates selected for the update
budget; they are not independently retuned for equal work. Flat baselines have
more tuning candidates, so tuning compute is not equal. Three reporting seeds
are exploratory evidence, not a significance test or a universal ranking.
Muon on vector-shaped problems has degenerate matrix geometry and is not the
hybrid Muon recipe used by the LLM runner.

To run a fresh benchmark and generate its reports, start from the repository root:

```bash
# The subshell returns to the repository root when it finishes.
(
  cd wrapper
  python -m pip install -e '.[bench,test]'
  python -m benchmarks.suite --output benchmarks/results/my_run --steps 100 --seeds 3
  python -m benchmarks.report benchmarks/results/my_run --site ../website
  python -m benchmarks.make_notebook --results benchmarks/results/my_run
)
```

Use a fresh output directory: the suite refuses to overwrite an existing
manifest. Report generation writes website assets; it does not publish the
website. The earlier endpoint-only and coarse-grid runs remain available under
[`endpoint_reference/`](wrapper/benchmarks/results/endpoint_reference/) and
[`coarse_reference/`](wrapper/benchmarks/results/coarse_reference/). Consult the
protocol before comparing these different experimental settings. Neither TACO
nor `pade3_21` is part of the reported reference atlas.

---

## Paper experiment package

The command-line package for the large-model experiments: four ZFO
local models, AdamW, AdamW with cosine decay, Prodigy, MeZO, ZFO-Seq, and Muon.
The original reproduction path consists of these files:

| File | Purpose |
| --- | --- |
| [`paper/run.py`](paper/run.py) | Configuration, paper grids, training, evaluation, local logging, aggregation |
| [`paper/zfo.py`](paper/zfo.py) | Taylor/Padé models, step selection, safeguards, parameter updates |
| [`wrapper/ZFO/_search.py`](wrapper/ZFO/_search.py) | Shared mathematical selectors, imported by both the library and legacy driver |
| [`paper/baselines.py`](paper/baselines.py) | Optimizers, cosine schedule, MeZO, sequential ZFO refinement |
| [`paper/prep.py`](paper/prep.py) | Models, datasets, prompts, reward functions, split snapshots |
| [`paper/requirements.txt`](paper/requirements.txt) | Pinned Python package dependencies |
| [`paper/README.md`](paper/README.md) | Complete option reference, reproduction commands, implementation notes |

No account-specific tracker or remote experiment dashboard is required. Install
`../wrapper` before running experiments. All results are written locally; the
mathematical selectors are shared with the wrapper.

**Working directory:** every command in the following paper usage and
reproduction sections runs from `paper/`. From the repository root:

```bash
cd paper
```

The installation instructions below create a separate Python 3.12 environment
and install both the pinned paper dependencies and `../wrapper`. Do not use the
wrapper's minimal dependency installation alone for the full LLM runner.

### Usage and hyperparameters

The command-line spelling uses underscores. Both `--method=value` and
`--method value` work. Inspect the fully resolved settings before a run:

```bash
python run.py --preset qwen_gsm8k --method pade3 --dry_run
python run.py --help
```

Configuration precedence is: base defaults, `--config` file, `--preset`, then
explicit command-line values. A preset chooses the model, dataset, learning
rate, training configuration, subset limits, and method-specific bound.
`adamw_cosine` always selects the cosine schedule. Run manifests store the
resolved settings, not just overrides.

#### Methods

| `--method` | Behavior |
| --- | --- |
| `adamw` | Full-parameter AdamW with a constant learning rate by default |
| `taylor2` | Quadratic Taylor model, two symmetric probes |
| `taylor3` | Cubic Taylor model, two symmetric probes and backward derivative |
| `pade2` | Padé [1/1] model |
| `pade3` | Padé [1/2] model |
| `adamw_cosine` | AdamW with linear warmup followed by cosine decay |
| `prodigy` | Adaptive optimizer with its own distance/step estimate |
| `mezo` | Two Gaussian directional probes of a fixed-rollout RLVR objective |
| `zfo_seq` | Initial Taylor/Padé proposal followed by evaluated refinements |
| `muon` | Muon on hidden 2D weights; AdamW on embeddings, output head, and remaining parameters |

`muon` is an additional comparison, with no claimed target score in the paper.
Its matrix learning rate must be tuned independently; it is not the selected
AdamW rate. All methods here use full parameter training and the GRPO policy.
There is no quantization, adapter training, gradient clipping, or gradient
accumulation in the FO/ZFO reproduction path.

#### Search interval and the roles of `lr` and `bound`

Let `theta_hat` be the tentative optimizer parameters and
`d = theta_hat - theta` the realized displacement. ZFO selects a multiplier
`a` in `[0, beta]`, where `beta = bound`, and applies `theta + a*d`.
The physical search radius is **`R = beta*||d||`**. This is the convention used
in the large-model experiments (Appendix C.1.1).

For nonzero `d`, writing `s = a*||d||` gives the equivalent expression
`theta + s*(d/||d||)`, with `s` in `[0, R]`. This is a change of units for
one interval; the implementation uses multiplier coordinates throughout.

The learning rate `alpha = lr` enters when the base optimizer constructs `d`;
`beta` scales the search interval afterward. At a fixed parameter, gradient,
and optimizer state, if the update is `d(alpha) = alpha*u` with `u` independent
of `alpha`, equal products `alpha*beta` give the same physical interval in
exact arithmetic. They need not do so in finite precision: storing the
tentative parameters and subtracting the originals can round different
coordinates differently, especially in BF16/FP16. Scaling the resulting `d`
with `beta` cannot recover a component already rounded away. This mechanism
is not specific to AdamW, and normalizing the realized displacement does not
remove it. It is a possible source of matched-product differences, not proof
that rounding alone explains all differences between training runs. Absolute
selector thresholds, probe arithmetic, and diverging training states can also
affect the selected step.

`perturbation_scale` is the physical probe distance `epsilon`: probes use
multipliers `+epsilon/||d||` and `-epsilon/||d||`. Probes can lie outside the
nonnegative search interval. The selected multiplier remains in `[0, bound]`.
Optimizer moments advance on each tentative update even when the selected
multiplier is zero.

Older command examples may contain `--search_type`. The current CLI no longer
exposes this option; it uses optimizer-displacement multiplier coordinates.
Saved configurations with the old default can load, while alternate-coordinate
configurations require their original source or an explicit migration. Removing
that field opts into the current selector and may change numerical decisions.

#### Interior candidates: `search_inside`

| Value | Taylor2 / Taylor3 | Padé3 | Quadratic safeguard |
| --- | --- | --- | --- |
| `auto` **(default)** | Endpoints and real stationary points | Endpoints | Can select an interior point |
| `false` | Endpoints only | Endpoints only | Endpoints only |
| `true` | Endpoints and real stationary points | Endpoints and real stationary points, after pole checks | Can select an interior point |

```bash
python run.py --preset qwen_svamp --method pade3 --search_inside=true
python run.py --preset qwen_svamp --method taylor3 --search_inside=false
```

The main LLM Padé3 experiments used endpoint selection (`search_inside=auto`), with interior steps possible only through the quadratic fallback triggered by poles or other numerical safeguards.

Padé3 uses the [1/2] rational form. Seven interior denominator checks detect
near-poles; they are not step-selection candidates. `false` also restricts the
quadratic safeguard to endpoints; `true` adds the analytic roots of the rational
derivative without extra model forwards.

Padé2 has no isolated interior stationary point when its denominator is regular,
so it remains an endpoint selector. Its historical degeneracy and denominator
handling are preserved. Taylor and Padé tie-breaking also follow the original
implementations; flat Taylor models prefer zero, while endpoint ties in Padé
prefer the upper bound.

For **ZFO-Seq**, `auto` preserves the separate sequential implementation,
including its 513-point Padé3 initializer grid. Explicit `true`/`false` changes
which candidates the **initial local model** considers; sequential refinements
can still evaluate and select interior points. The final sequential choice is
the best evaluated point in the interval, including the initial zero step.

#### Model, data, training, and sampling options

Defaults below are the base configuration; presets can override the marked
training and data settings.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen` | Model alias below, or a public model repository ID |
| `--dataset` | `gsm8k` | `gsm8k`, `math`, `svamp`, `asdiv`, `openbookqa`, `arc_challenge`, `strategyqa`, `folio`, `codah` |
| `--method` | `taylor2` | Method from the table above |
| `--seed` | `100` | Python, NumPy, CPU, and CUDA seed |
| `--epochs` | `10` | Maximum complete training epochs; preset dependent |
| `--batches` | `4` | Number of prompts per minibatch; preset dependent |
| `--group_size` | `4` | Sampled responses per prompt, at least 2; preset dependent |
| `--maxlength` | `512` | Maximum prompt tokens; preset dependent |
| `--max_new_tokens` | `192` | Maximum generated response tokens |
| `--subset` | `1000000` | Training prefix cap, clamped to available data; preset dependent |
| `--eval_subset` | `1000000` | Evaluation prefix cap; also the MATH held-out size; preset dependent |
| `--asdiv_eval_fraction` | `0.2` | Fraction of distinct ASDiv problems held out if only one source split is available |
| `--asdiv_split_seed` | `0` | ASDiv partition seed, independent of training seed; shared across methods/models |
| `--temperature` | `0.8` | Training sampling temperature |
| `--top_p` | `0.95` | Training nucleus-sampling probability |
| `--kl_beta` | `0.02` | KL penalty coefficient, distinct from search `bound` |
| `--early_stop_patience` | `3` | Stop after this many exactly equal consecutive epoch scores; `0` disables |
| `--model_revision` | unset | Model commit or ref; unset uses the checked pin for the four known models |
| `--dataset_revision` | unset | Dataset commit or ref; unset uses the checked pin in `prep.py`; FOLIO uses its public source repository |
| `--data_dir` | unset | Replay a prepared directory containing `train.jsonl`, `eval.jsonl`, `data.json` |

Evaluation occurs after each epoch, using one greedy response per prompt. The
reported result is the **last** evaluation, not the best checkpoint. There is
no evaluation before epoch 1. Early stopping counts the first occurrence as 1:
three consecutive identical scores stop after the third occurrence.

| Model alias | Repository ID |
| --- | --- |
| `qwen` | `Qwen/Qwen2.5-Math-1.5B` |
| `phi2` | `microsoft/phi-2` |
| `gemma2_2b` | `google/gemma-2-2b` |
| `llama32_1b` | `meta-llama/Llama-3.2-1B` |

Gated models require access to their public distribution and an `HF_TOKEN` in
the environment. Tokens are not written to manifests.

#### Optimizer and search options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lr` | `5e-6` | AdamW learning rate; MeZO update rate; auxiliary AdamW rate for Muon. Preset dependent. Prodigy uses `prodigy_lr` instead. |
| `--bound` | `10` | Dimensionless upper multiplier `beta`; preset/method dependent; unused by plain optimizers and MeZO |
| `--perturbation_scale` | `1e-3` | ZFO physical probe distance; MeZO unnormalized Gaussian perturbation scale |
| `--search_inside` | `auto` | Candidate policy described above |
| `--beta1`, `--beta2` | `0.9`, `0.999` | AdamW and Prodigy moment coefficients |
| `--adam_eps` | `1e-8` | AdamW and Prodigy denominator epsilon |
| `--weight_decay` | `0.01` | AdamW, Prodigy and Muon decay |
| `--lr_type` | `constant` | `constant` or `warmup_cosine` for AdamW-backed methods |
| `--warmup_ratio` | `0.1` | Fraction of the planned epoch-times-batch update count used for warmup |
| `--prodigy_lr` | `1.0` | Prodigy adjustment scale |
| `--d_coef` | `1.0` | Prodigy adaptation coefficient |
| `--d0` | `1e-6` | Initial Prodigy distance estimate |
| `--muon_lr` | `0.02` | Hidden matrix learning rate, separate from `lr` |
| `--muon_momentum` | `0.95` | Muon momentum, with Nesterov enabled |
| `--muon_ns_steps` | `5` | Muon Newton-Schulz iterations |
| `--zo_weight_decay` | `0.0` | MeZO decay; excludes bias and normalization parameters |
| `--zo_max_update_norm` | unset | Optional MeZO update norm limit; unset leaves updates unclipped |
| `--k` | `5` | ZFO-Seq extra objective calls including its first two symmetric probes; at least 2 |
| `--initial_model` | `p3` | ZFO-Seq initializer: `t2`, `t3`, `p2`, `p3` |
| `--tol` | unset | Sequential duplicate-point tolerance in multiplier units; unset means `0.001*bound` |

Prodigy uses decoupled decay and no external schedule. Muon uses the PyTorch
`match_rms_adamw` scaling rule, Nesterov momentum, default Newton-Schulz
coefficients and epsilon; tied embeddings/output weights are assigned to AdamW
exactly once. Muon and MeZO also require a constant external schedule here.
See the [Muon API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
and [Prodigy implementation](https://github.com/konstmish/prodigy).

Cosine follows the historical `LambdaLR` indexing: warmup uses
`min(total_steps, max(1, round(warmup_ratio*total_steps)))` steps, and the schedule
advances after the optimizer update. Even `warmup_ratio=0` retains one warmup
step. Early stopping can terminate before the schedule reaches its end.

The old `alpha_fd_min`/`alpha_fd_max` options did not affect the supplied surrogate
trainer. They are omitted rather than exposed as ineffective controls.

#### Execution and output options

| Option | Default | Meaning |
| --- | --- | --- |
| `--preset` | unset | One `model_dataset` row from the configuration table below |
| `--config` | unset | Replay a saved resolved `config.json` |
| `--suite` | unset | `table1`, `table2`, `table10`, `table11`, or `muon`; lists configurations by default |
| `--run_index` | unset | Zero-based suite entry to list or execute |
| `--execute` | off | Execute a suite, one fresh process per run; stops on failure |
| `--dry_run` | off | Print settings without loading models or training libraries |
| `--output` | `results` | Output parent directory; an existing run directory is never overwritten |
| `--prepare_only` | off | Download/select data and write its snapshot, without loading model weights |
| `--summarize` | unset | Aggregate completed runs below a given directory |
| `--self_test` | off | Run local numerical tests without downloading models or datasets |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`; auto prefers CUDA |
| `--dtype` | `auto` | `float32`, `float16`, `bfloat16`, or auto: CUDA bf16 if supported, otherwise fp16; CPU fp32 |
| `--attn_implementation` | `auto` | Model default, `eager`, `sdpa`, or `flash_attention_2`; the latter requires its own compatible installation |
| `--deterministic` | off | Enable deterministic tensor algorithms; unsupported kernels raise an error |
| `--save_model` | off | Save final tensor weights, model configuration and tokenizer |

Boolean switches also accept `--no-deterministic` and `--no-save_model`.
`--search_inside` takes a value; it is not a valueless switch. Suite grids are
fixed: use a single preset or a saved configuration for modified experiments.
`--help` and `--dry_run` work with the Python standard library alone.

### Reproducing the experiments

#### Install

`requirements.txt` pins the package versions used in the packaging checks.
Keep installed environments and generated outputs separate from the experiment
source files. From `paper/`, use Python 3.12 in an isolated environment:

```bash
python3.12 -m venv ../zfo-env
source ../zfo-env/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ../wrapper
python run.py --self_test
```

Install the CUDA build of the same PyTorch release suitable for the training
machine, using the [official installation instructions](https://pytorch.org/get-started/previous-versions/).
The original GPU software lockfile and exact model/data revisions were not
available. The four known models and nine datasets have immutable revision
pins in `prep.py`, checked during packaging. These are tested package dependencies
and asset pins, **not a recovered historical environment**. Large-model runs need a CUDA GPU with room for the policy,
frozen reference, optimizer state, and ZFO parameter snapshots. Full-model
memory requirements are much larger than inference requirements.

#### Paper configuration table

The presets transcribe Tables 4-7. Seeds are `100`, `101`, `102` for Tables 1,
2 and 10. Each tuple lists bounds in the order **Taylor2, Taylor3, Padé2, Padé3**.

| Preset | FO/ZFO LR | MeZO LR | Bounds | Type |
| --- | --- | --- | --- | --- |
| `qwen_gsm8k` | 5e-6 | 1e-7* | 10, 5, 10, 10 | a |
| `qwen_math` | 5e-6 | 1e-6 | 5, 5, 5, 5 | c |
| `qwen_svamp` | 1e-5 | 5e-7 | 3, 5, 3, 3 | a |
| `qwen_asdiv` | 5e-6 | 5e-7 | 5, 5, 5, 5 | c |
| `qwen_openbookqa` | 1e-5 | 1e-7 | 10, 5, 5, 5 | a |
| `phi2_svamp` | 5e-6 | 5e-7 | 3, 5, 5, 3 | a |
| `phi2_asdiv` | 5e-6 | 1e-7 | 3, 3, 5, 3 | d |
| `phi2_openbookqa` | 5e-6 | 1e-7 | 3, 5, 3, 3 | d |
| `gemma2_2b_svamp` | 1e-6 | 5e-9 | 3, 3, 3, 3 | b |
| `llama32_1b_gsm8k` | 5e-6 | 1e-7 | 3, 3, 3, 3 | a |
| `llama32_1b_asdiv` | 5e-6 | 5e-8 | 5, 5, 3, 3 | c |
| `llama32_1b_openbookqa` | 1e-6 | 1e-7* | 5, 5, 5, 3 | b |
| `qwen_arc_challenge` | 1e-5 | - | 3, 5, 3, 3 | c |
| `qwen_strategyqa` | 1e-5 | - | 5, 5, 3, 5 | c |
| `qwen_folio` | 1e-5 | - | 3, 3, 3, 3 | c |
| `qwen_codah` | 1e-5 | - | 5, 5, 5, 5 | c |

\* The paper reports `ALL` for these MeZO cells. `1e-7` is an explicit
representative setting, not a uniquely recovered winning rate.

| Type | Epochs | Batch | Group | Prompt limit | Output limit |
| --- | --- | --- | --- | --- | --- |
| a | 10 | 4 | 4 | 512 | 192 |
| b | 8 | 2 | 2 | 384 | 192 |
| c | 8 | 8 | 2 | 384 | 192 |
| d | 5 | 4 | 2 | 256 | 192 |

The main and sub-study presets use the full named splits where available.
MATH uses 512 training and 256 evaluation rows. ASDiv uses an 80/20 split
with partition seed 0: 1,844 training and 461 evaluation examples. Duplicate
problem texts stay in the same partition. Training seeds 100/101/102 share
the same evaluation examples, and subset caps apply after partitioning.
The sequential comparison uses prefix caps of 512 and 256, including the
CODAH split's smaller available held-out set. Subset size and exact row
order matter.

#### Run one cell or a full grid

```bash
# One cell, three seeds.
for seed in 100 101 102; do
  python run.py --preset qwen_gsm8k --method taylor2 --seed "$seed" --output ../zfo-results/table1
done

# Inspect a complete grid, then launch it.
python run.py --suite table1 --dry_run
python run.py --suite table1 --execute --output ../zfo-results/table1

# Or execute one indexed job, suitable for a scheduler array.
python run.py --suite table1 --run_index 0 --execute --output ../zfo-results/table1
```

Choose either individual jobs or the complete suite for a given output parent;
already-existing runs cause an error instead of being silently reused or
replaced. To rerun a cell, use a new output parent. No run resume is implemented.

| Suite | Contents | Jobs |
| --- | --- | --- |
| `table1` | 12 model/dataset settings x 6 methods x 3 seeds | 216 |
| `table2` | 4 additional Qwen datasets x 5 FO/ZFO methods x 3 seeds | 60 |
| `table10` | 8 Qwen datasets x AdamW-cosine/Prodigy x 3 seeds; excludes MATH | 48 |
| `table11` | ZFO-Seq on CODAH, ARC-Challenge, SVAMP; seed 100 | 3 |
| `muon` | 8 Qwen datasets x 3 seeds; additional untuned comparison | 24 |

```bash
python run.py --suite table2 --execute --output ../zfo-results/table2
python run.py --suite table10 --execute --output ../zfo-results/table10
python run.py --suite table11 --execute --output ../zfo-results/table11
python run.py --suite muon --execute --output ../zfo-results/muon
```

The Table 11 suite fixes Qwen, type c, LR `1e-5`, bound `3`, epsilon `1e-3`,
`k=5`, initializer `p3`, seed `100`, and subset caps `512/256`. It reproduces
the ZFO-Seq column only; Armijo, BB, Polyak and PLS are outside this package.
The classical/toy figures and additional appendix ablations are also outside
these large-model grids.

#### Preserve data and repeat a run

```bash
python run.py --preset qwen_gsm8k --method taylor2 --prepare_only --output ../zfo-prepared
```

Every data preparation or training run saves its exact ordered selected rows.
To use a prepared run, replace `PREPARED_RUN` below with its directory. It must
match `dataset`, `seed`, `subset`, and `eval_subset`; content hashes are checked.

```bash
python run.py --config PREPARED_RUN/config.json --data_dir PREPARED_RUN --output ../zfo-results/frozen
```

For a trained run, `environment.json` records the resolved model commit and
`data.json` records the dataset commit. To repeat it, replace `MODEL_COMMIT`
with that recorded model SHA and `RUN` with that run directory:

```bash
python run.py --config RUN/config.json --data_dir RUN \
  --model_revision MODEL_COMMIT --output ../zfo-results/repeat
```

Presets use the checked immutable pins in `prep.py`; they do not claim to
identify the historical upstream revisions used in every paper run. Custom
model repositories without a packaged pin resolve their requested ref (or
`main`) once at load time. Save snapshots and use the recorded commit plus the
recorded software versions for later reruns.
The data loader fails if a designated dataset/split is unavailable; it never
silently switches to a different mirror.

#### Outputs and aggregation

A training run creates:

- `config.json`: resolved hyperparameters, without the local prepared-data path.
- `environment.json`: package versions, device/dtype, model commit, and source-file hashes.
- `data.json`, `train.jsonl`, `eval.jsonl`: exact selected examples, provenance and hashes.
- `metrics.jsonl`: every update and epoch evaluation, including selected steps and fallback flags.
- `result.json`: completion status, final accuracy, stopping epoch, elapsed time and memory.
- `model/`: final tensors and tokenizer, only with `--save_model`.

```bash
python run.py --summarize ../zfo-results/table1 > table1_summary.jsonl
```

Aggregation groups matching configurations, source hashes, resolved revisions,
dtypes and package versions. It rejects duplicate seeds and reports seed
coverage, the mean percentage, and both sample and population standard
deviations. A three-seed claim requires `paper_seeds_complete: true`.
Table 11 intentionally uses one seed. Accuracy in per-run files is a fraction;
summary means and standard deviations are in percentage points.

#### Implementation fidelity and limits

The training and evaluation implementation has the following properties:

1. The GRPO backward pass supplies the update direction and first-order
   coefficient. In maximization notation, its objective is
   `F(theta) = mean(A * sequence_logp(theta)) - kl_beta * K(theta)`, where
   `K` is the mean signed token log-ratio estimate. The symmetric probes use
   `Q(theta) = mean(frozen_rewards) - kl_beta * K(theta)` on the same sampled
   sequences. The local step-selection model combines the slope from `F`
   with values from `Q`. Freezing rewards and advantages does not freeze
   `sequence_logp(theta)`, so `F` and `Q` remain distinct objectives. Guarantees
   requiring a slope and probe values from one scalar objective do not directly
   cover this mixed construction.
2. FO/ZFO masks include nonpadding prompt and completion tokens. With left
   padding, the historical decoder slices each completion at its unpadded
   prompt length, which can include part of the prompt. Training rollouts run
   in training mode; probes and greedy evaluation use evaluation mode. When
   dropout is active, this mode change can affect forward values before any
   parameter update; fixed rollouts do not imply a shared dropout realization.
   The pinned Phi-2 configuration has residual dropout probability 0.1.
   These details are retained because changing them can change rewards and updates.
3. ASDiv uses the 80/20 data protocol described above. `data.json` records
   the partition seed, counts, and content hashes. MATH uses a seeded shuffled
   training-split holdout; CODAH uses a seeded 90/10 split. StrategyQA prompts
   include the provided facts and decomposition hints.
4. MeZO follows the available MeZO source, which has a completion-only mask,
   padded-width decoding and a nonnegative k3 KL estimate. Its behavior is not
   identical to the FO/ZFO probe objective. Its in-place symmetric perturbation
   restoration also inherits finite-precision rounding.
5. Padé3 has pole and near-pole checks with quadratic fallback. Padé2 retains
   the original endpoint denominator handling rather than introducing a new
   interval-wide pole safeguard. The original absolute thresholds are retained in multiplier coordinates;
   they are not guaranteed to be invariant under rescaling.
6. The selected configurations are transcribed from the paper and checked
   against available launchers, not reconstructed from a complete immutable
   manifest of the exact three runs behind every published cell. The Qwen/MATH
   preset uses type c (8 epochs, batch size 8, group size 2, and prompt limit
   384), matching the revised Table 6 and the available MATH launcher.

Accordingly, this is an executable reproduction package with preserved default
numerical logic and explicit settings, **not a certification that fresh runs
will match the published means**. Historical model/data commits, the exact
GPU environment, and the complete selected-run manifest are still needed for
that certification.

#### Validation

The package includes `--self_test` for data partitioning, endpoints, interior roots, physical
probe spacing and search radius, the sequential budget, cosine scheduling, and grid
sizes. Additional packaging checks compare scalar selectors, prompts, verifiers,
schedules and generation against the supplied source, and exercise tiny local
training runs for every method. Forty-eight checks passed, including exact final
tensor agreement with the original training loop on seven small FO/ZFO cases.
All nine dataset loaders were checked against their pinned sources. A command-line
run with a small downloaded model also completed training, evaluation and result
aggregation. These checks do not substitute for full GPU reruns of the paper grids.

---

## Project website and function playground

**[Visit the live ZFO website](https://nizswan.github.io/Zeroth-First-Order-Framework/)**
for the method illustration, configuration builder, paper PDF, and function
playground. The site source is in [`website/`](website/), rather than `docs/`.

The playground accepts functions of `x`, arithmetic, powers, parentheses,
`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pi`, and `e`. It compares Taylor2,
Taylor3, Padé2 [1/1], and Padé3 [2/1] for a single illustrative step. Its Padé3
form corresponds to the wrapper's `pade3_21`; the paper experiment runner's
`pade3` is [1/2]. These labels should not be treated as interchangeable.

Expressions are parsed as math rather than evaluated as JavaScript. The initial
slope is computed by automatic differentiation and curvature is estimated from
two probes. The chart distinguishes local predictions from actual function
values; it is a visualization of a local step, not a full training experiment.

To preview the website, run this from the repository root:

```bash
python -m http.server 8765 --directory website
```

Then open [http://localhost:8765](http://localhost:8765).
The [Pages workflow](.github/workflows/pages.yml) publishes `website/` when site
changes are pushed to `main`, or when manually dispatched. The public site URL
remains the same after the repository reorganization. See the
[website guide](website/README.md) for figure generation and saved-asset details.

## Repository checks

The wrapper, paper runner, and website have separate checks. From the repository
root, install the wrapper test/benchmark extras before running the Python tests:

```bash
python -m pip install -e './wrapper[test,bench]'

# Wrapper and benchmark checks (run within wrapper/ for its benchmark modules).
(cd wrapper && python -m pytest -q)

# Paper selector and runner checks; no large-model training is launched.
python paper/run.py --self_test
(cd paper && python -m pytest -q tests)

# Website syntax and numerical/input checks (requires Node.js).
node --check website/site.js
node --test website/tests/test_site_playground.cjs
```

The [library workflow](.github/workflows/tests.yml) also builds a source
distribution and wheel and runs the installed wheel's supervised example outside
the checkout. The [Pages workflow](.github/workflows/pages.yml) runs the
playground checks before deployment.

These automated checks cover package behavior and selected numerical cases;
they do not certify GPU compatibility for every environment, monotonic loss
improvement, or fresh reproduction of the paper's reported means.
