# Use ZFO in a training loop

Install directly with
`python -m pip install "git+https://github.com/nizswan/Zeroth-First-Order-Framework.git#subdirectory=wrapper"`,
or from a checkout with `python -m pip install -e .`. Requires Python 3.10+
and PyTorch 2.9+. The package is
named `zeroth-first-order`; the public import is `from ZFO import ZFO` (or
`import ZFO` followed by `ZFO.ZFO(...)`). No PyPI release is implied.
The only core dependency is PyTorch. LLM reproduction dependencies remain in
[`paper/requirements.txt`](../../paper/requirements.txt).

```python
import torch
from ZFO import ZFO

model = torch.nn.Linear(8, 1).double()
base = torch.optim.Adam(model.parameters(), lr=1e-2)
optimizer = ZFO(base, model=model, epsilon=1e-3, method="taylor2",
                interior_search="auto", beta=3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base, T_max=100)

for x, y in loader:
    # Freeze the batch outside the closure. Do not call backward in it.
    loss = optimizer.step(lambda: (model(x) - y).square().mean())
    scheduler.step()
```

The closure returns the same scalar objective for the gradient and probes.
ZFO makes one ordinary optimizer update to obtain `d = theta_hat - theta`,
fits a local model with symmetric forward probes, and applies
`theta_new = theta + a*d`, where `0 <= a <= beta`. `a=1` is the ordinary update;
`a=0` leaves parameters in place. Optimizer momentum/moments still advance
exactly once when `a=0`. Weight decay is included in `d` and therefore scaled
with the rest of the displacement. This is a choice inherited from the research
implementation, not a claim of equivalence to a conventional LR scheduler.

| Argument | Default | Meaning |
| --- | --- | --- |
| `optimizer` | required | An existing closure-free PyTorch optimizer; SGD, Adam, AdamW, matrix-only Muon, or the included matrix-only TACO |
| `epsilon` | `1e-3` | Physical parameter-space probe distance, not a multiplier or Adam's numerical epsilon |
| `method` | `"taylor2"` | `taylor2`, `taylor3`, `pade2` ([1/1]), `pade3` ([1/2]), or `pade3_21` ([2/1]) |
| `interior_search` | `"auto"` | `True`, `False`, or `"auto"`; details below |
| `beta` | `1.0` | Region multiplier of the realized optimizer displacement; benchmark uses `3.0` explicitly |
| `lr` | `None` | Optional override of all base-optimizer group rates; omitting it preserves group-specific rates |
| `model` | `None` | Pass the model to restore buffers around probes, especially with BatchNorm |

For the general benchmark atlas, pass **`interior_search=True`** explicitly:
this allows Padé3 to compare interior stationary points with the endpoints.
The atlas uses this setting throughout; the original RLVR runner retains auto.

`auto` preserves the research settings: Taylor stationary points, Padé endpoints,
and possible interior quadratic fallbacks. `True` additionally enables Padé3
stationary points for both [1/2] and [2/1] after their denominator checks. `False` restricts every model
and fallback to endpoints. Padé2 has no isolated stationary point for a regular
denominator. The adapter preserves the historical Padé2 degeneracy/pole handling;
it does not silently substitute the differently safeguarded sequential variant.

### Padé [2/1] library extension

`method="pade3_21"` uses the same objective value, directional derivative, and
symmetric probes as `pade3`. It is an additional local model, not a reported
paper or benchmark setting. The original `pade3` continues to mean [1/2].

Writing the local coefficients as `c0=f0`, `c1=d1`, `c2=d2/2`, `c3=d3/6`,
the [2/1] model is `(a0+a1*x+a2*x²)/(1+b1*x)`, with `b1=-c3/c2`,
`a0=c0`, `a1=c1+c0*b1`, and `a2=c2+c1*b1`. No additional probes are needed.
Invalid coefficients or a pole in the interval trigger a quadratic fallback;
the shared near-zero-curvature safeguard uses the existing endpoint rule.
`auto` compares endpoints; `True` also compares interior stationary points.

### Search interval

Let `d = theta_hat - theta` be the realized optimizer displacement. ZFO searches
`a ∈ [0, beta]` and applies `theta + a*d`, matching the large-model experiment
convention in Appendix C.1.1. The physical radius is `R = beta*||d||`.
For nonzero `d`, the distance coordinate `s = a*||d||` describes the same
interval as `theta + s*d/||d||`, `s ∈ [0, R]`; it needs no separate setting.
The norm includes the whole displacement, including when parameter groups
have different learning rates. Diagnostics report the selected multiplier.

Learning rate and `beta` act at different stages. If, at a fixed optimizer
state and gradient, `d(alpha) = alpha*u` for an `alpha`-independent direction,
equal `alpha*beta` gives the same physical interval in exact arithmetic.
Finite-precision parameter updates can break this proportionality before
ZFO forms its search line. Normalizing the realized displacement cannot
recover coordinates rounded away. This effect is not specific to AdamW;
its magnitude depends on the update, parameter values, and precision. The
adapter requires float32/float64 parameter storage; the original LLM path
also supports BF16/FP16. Absolute selector thresholds and probe arithmetic
can additionally change numerical decisions. See the [repository explanation](../../paper/README.md#search-interval-and-the-roles-of-lr-and-bound).

Older default checkpoints load normally. A checkpoint requesting the removed
alternate coordinate mode is rejected before loading optimizer state; use its
original source to reproduce it, or explicitly remove its saved `search_type`
field to adopt this selector, whose numerical decisions may differ.

**The selected point is not verified with another forward.** Local model error,
Padé poles, minibatch overfitting, or finite-difference cancellation can produce
a worse actual loss. A usual step costs three forwards and one backward; a
zero displacement needs only the original forward. Probe points can be outside
the final search interval. Read `optimizer.last_step` for the multiplier,
forward count, effective group rates, and fallback diagnostics. A fallback
flag reports a numerical model fallback, not a successful acceptance test.

## Supported training patterns

- Full-batch and minibatch supervised regression/classification with arbitrary
  differentiable scalar losses; data loading stays in the training loop.
- Single-device CPU, CUDA, or MPS, with float32/float64 parameter storage.
  The included numerical benchmarks use CPU float64; accelerator behavior
  requires validation on the intended hardware. Double precision helps tiny
  finite differences; float32 may require a larger epsilon.
- Multiple parameter groups, frozen parameters, native weight decay, and base
  optimizers with `maximize=True`. All groups must agree on maximization.
- Torch dropout with repeated RNG states; mutable model buffers are restored
  around probes when `model` is supplied. The final RNG and buffers reflect one
  training forward. Python/NumPy randomness and external side effects are not
  replayed: precompute augmentations and sampled rollouts outside the closure.
- Differentiable fixed-rollout policy objectives can use the same interface.
  Return the **negative** reward surrogate for a minimizing optimizer, or use
  a maximizing optimizer. Rewards, rollout sampling, and KL construction remain
  caller responsibilities. The existing [`paper/run.py`](../../paper/run.py) remains the LLM reproduction path.

This first adapter does not implement AMP/GradScaler, distributed/sharded
training, compiled/captured steps, sparse gradients, LBFGS, or gradient
accumulation. It owns `zero_grad` and `backward`; do not use it as a drop-in
trainer optimizer expecting precomputed gradients. Muon accepts 2D parameters
only; the examples use matrix parameters, while the original LLM driver uses
its own Muon/AdamW bundle. A multi-optimizer hybrid adapter is not included.

## TACO library adapter

`from ZFO import TACO` exposes a matrix-only PyTorch optimizer implementing
normalized gradient momentum (`momentum=0.95`) and a signed top-1 update in
each column, based on the [TACO v8.2 notebook](https://anonymous.4open.science/r/iclr27-TACO-923D/Test_TACO_FO_v8.2.ipynb).
It is compatible with ZFO and ordinary PyTorch schedulers/checkpoints:

```python
from ZFO import ZFO, TACO

# All trainable parameters must be logical 2D matrices for this example.
params = [p for p in model.parameters() if p.requires_grad]
base = TACO(params, lr=3e-5, momentum=0.95)
optimizer = ZFO(base, model=model, method="pade3_21")
```

This adapter uses dense history in the parameter dtype (float32/float64), with
updates during `step()`. It does not reproduce the notebook's compressed E4M3
hk16 history, gradient-release hooks, auxiliary AdamW, or architecture-specific
routing. Its default `spectral_scale=1` matches the ordinary-matrix rule;
set group-specific `spectral_scale=fan_out/fan_in` for feed-forward matrices
when applying that reference policy. Split fused projections into logical
matrices before using this adapter. The source notebook remains the reference
for TACO paper experiments. The existing ZFO benchmark atlas has no TACO runs.

## Checkpoints and errors

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

Also save device RNG, sampler position, and any other training state your
application needs. The adapter restores parameters, gradients, optimizer state,
buffers, and Torch RNG if a closure/probe raises or produces a nonfinite value.
That transaction takes parameter and optimizer-state snapshots each step: it
adds memory and Python overhead. It is intended as a transparent general-purpose
reference implementation; profile large models before adopting it in production.

Checkpoint loading validates the saved settings and parameter groups; an invalid
checkpoint leaves the current optimizer and wrapper state intact.

Run `python examples/supervised.py` for a complete example and
`python examples/supervised.py --method pade3_21 --interior-search true`
for the additional Padé variant. The example also accepts `--optimizer`, `--lr`,
`--beta`, and `--epsilon`. Install `.[test,bench]` and run `python -m pytest -q`
for behavioral checks.
