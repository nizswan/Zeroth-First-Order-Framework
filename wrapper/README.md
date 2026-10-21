# Reusable PyTorch wrapper

ZFO selects a step along an optimizer's proposed direction using Taylor or Padé
models and two symmetric objective probes. This repository now includes a
reusable PyTorch adapter, a twelve-problem benchmark atlas, and a GitHub Pages
site alongside the original large-model experiment runner.

| Start here | What you get |
| --- | --- |
| [Library guide](docs/library.md) | `from ZFO import ZFO`, supervised training, schedulers, checkpoints, and limitations |
| [Complete example](examples/supervised.py) | A small supervised regression run with Adam + ZFO |
| [Benchmark notebook](notebooks/benchmark_atlas.ipynb) | Repeated explanation → table → plots, including weak and unstable cases |
| [Benchmark protocol](benchmarks/README.md) | Independent LR scans, disjoint seed cohorts, raw evidence, and cost accounting |
| [Reference findings](benchmarks/FINDINGS.md) | Concrete examples of gains, regressions, failures, and compute tradeoffs |
| [Website](https://nizswan.github.io/Zeroth-First-Order-Framework/) | Configuration builder and interactive function playground |
| [Original CLI reference](../paper/README.md#usage-and-hyperparameters) | The paper-oriented LLM runner and reproduction settings |

## Install the library

Requires Python 3.10+ and PyTorch 2.9+. Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/nizswan/Zeroth-First-Order-Framework.git#subdirectory=wrapper"
```

For the examples, tests, and benchmark tools, use a checkout:

```bash
git clone https://github.com/nizswan/Zeroth-First-Order-Framework.git
cd Zeroth-First-Order-Framework/wrapper
python -m pip install -e .
python examples/supervised.py
```

```python
from ZFO import ZFO

# model, optimizer, inputs, targets, and loss_fn come from your training loop.
zfo = ZFO(optimizer, model=model, epsilon=1e-3, method="taylor2",
          interior_search="auto", beta=1.0)
loss = zfo.step(lambda: loss_fn(model(inputs), targets))
```

The closure returns a scalar loss; ZFO owns zeroing gradients and backward.
Attach any ordinary LR scheduler to the **base optimizer**, then call it after
`zfo.step`. Batch size, datasets, epochs, and model architecture stay outside
the adapter. The optional `lr=` overrides base group rates; omitting it preserves
them. `import ZFO` and `ZFO.ZFO(...)` also work.

This reference adapter supports single-device float32/float64 training and
includes all five local models:

| `method` | Local model |
| --- | --- |
| `taylor2` | Quadratic Taylor |
| `taylor3` | Cubic Taylor |
| `pade2` | Padé [1/1] |
| `pade3` | Padé [1/2], the original experiment variant |
| `pade3_21` | Padé [2/1], the additional library variant |

Both cubic Padé variants use the same two probes. Set `interior_search=True`
to include valid stationary points as well as endpoints. For example:

```bash
python examples/supervised.py --optimizer AdamW --method pade3_21 --interior-search true --beta 3
```

SGD, Adam, AdamW, matrix-only Muon, and the included matrix-only TACO adapter
are covered by the library tests. The wrapper does not provide a monotonic-loss
guarantee, AMP/distributed integration, or a hybrid Muon optimizer. See the
[API guide](docs/library.md) before adopting it in a larger trainer.

## Explore the evidence

```bash
python -m pip install -e '.[bench,test]'
python -m pytest -q
python -m benchmarks.suite --output benchmarks/results/my_run --steps 100 --seeds 3
python -m benchmarks.report benchmarks/results/my_run --site ../website
python -m benchmarks.make_notebook --results benchmarks/results/my_run
python -m http.server 8765 --directory ../website
```

The completed reference run covers SGD, Adam, and Muon × flat LR plus four ZFO
variants, with three tuning seeds and three distinct reporting seeds. Flat
baselines receive a denser 13-rate sweep; each ZFO variant retains seven rates,
with beta fixed at 3. Both grids span 1e-5 to 10. These
atlas runs explicitly enable `interior_search=True`, including Padé3 interior
stationary points; the original RLVR auto setting remains separate. These
small CPU experiments are exploratory. Selected grid boundaries, divergence,
and extra forward cost are explicit; the notebook does not claim a universal
winner. The [Pages instructions](../website/README.md) explain site deployment.
