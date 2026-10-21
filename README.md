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

ZFO uses Taylor or Padé models to choose a step along a first-order optimizer's
proposed direction. Choose the component you want to work with:

| Folder | Contents |
| --- | --- |
| [wrapper/](wrapper/) | Installable PyTorch wrapper, API guide, supervised examples, tests, and benchmark evidence |
| [paper/](paper/) | Paper experiment runner, datasets, baselines, pinned dependencies, and reproduction instructions |
| [website/](website/) | Project website, interactive playground, paper PDF, figures, and website tests |

## Use the wrapper

Requires Python 3.10+ and PyTorch 2.9+:

```bash
python -m pip install "git+https://github.com/nizswan/Zeroth-First-Order-Framework.git#subdirectory=wrapper"
```

```python
from ZFO import ZFO

# Keep your model, base optimizer, data loader, and scalar loss function.
zfo = ZFO(base_optimizer, model=model, method="pade3_21",
          interior_search=True, beta=3, epsilon=1e-3)
loss = zfo.step(lambda: loss_fn(model(inputs), targets))
```

The closure returns the loss; ZFO handles gradient zeroing and backward.
Taylor2, Taylor3, Padé [1/1], Padé [1/2], and Padé [2/1] are supported.
See the [wrapper guide](wrapper/docs/library.md) for schedulers, checkpoints,
supported training patterns, and limitations.

## Run the paper experiments

Start with [paper/README.md](paper/README.md) for installation, configurations,
and reproduction limits. Install the wrapper as described there before executing
the paper runner; both components share the same mathematical selectors.

## Explore the website

Visit the [project website](https://nizswan.github.io/Zeroth-First-Order-Framework/)
or preview it locally from the repository root:

```bash
python -m http.server 8765 --directory website
```

See [website/README.md](website/README.md) for asset generation and deployment.
The `.github/workflows/` directory contains the automated library checks and
Pages deployment configuration.
