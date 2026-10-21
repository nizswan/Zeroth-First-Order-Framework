"""Small, download-free objectives; all optimizer parameters are matrices.

Every method sees identical data, initialization, and minibatch order per seed.
Classical vectors use shape (1, d); Muon on these is a degenerate matrix case.
"""
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


DESCRIPTIONS = {
    "sphere": ("Isotropic quadratic", "classical", "A well-scaled convex control. A tuned constant step already works well; extra probes can be pure overhead."),
    "ill_conditioned": ("Ill-conditioned least squares", "classical", "A rotated linear system with condition number 10,000 in its Hessian. Tests scale sensitivity without nonconvexity."),
    "nonlinear_lse": ("Exponential curve fitting", "classical", "Fit amplitude, decay, and offset from a poor initial guess. Rapid curvature changes challenge a local polynomial or rational model."),
    "logistic": ("Logistic regression", "supervised", "Anisotropic features and stochastic binary labels. Tune on validation cross-entropy; report untouched test cross-entropy and accuracy."),
    "matrix_factorization": ("Low-rank matrix factorization", "classical", "Recover a rank-three matrix with two trainable factors. A nonconvex problem with genuine matrix geometry for Muon."),
    "rosenbrock": ("Rosenbrock valley", "classical", "A curved narrow valley with seed-perturbed starts. Local step selection cannot repair a poorly aligned direction."),
    "beale": ("Beale surface", "classical", "A two-variable nonlinear least-squares landscape. Includes seed-perturbed difficult starts and unstable learning rates."),
    "rastrigin": ("Rastrigin landscape", "stress", "Many local minima. Good local descent does not guarantee global progress; a deliberate limitation test."),
    "noisy_regression": ("Minibatch noisy regression", "supervised", "Noisy targets and batches of sixteen. Fixed-batch probes are coherent but may pick steps that generalize poorly to the next batch."),
    "moons": ("Two moons · tanh MLP", "supervised", "Nonlinear classification with a smooth 16-unit network, finite samples, and held-out validation/test data."),
    "xor": ("XOR · ReLU MLP", "stress", "A small piecewise-linear network crosses activation boundaries. The loss need not match a smooth Taylor model along the whole search interval."),
    "noisy_labels": ("Moons · 30% label noise", "stress", "A tanh classifier trained on corrupted labels in minibatches. Validation and test labels remain clean to expose overfitting."),
}


@dataclass
class Problem:
    parameters: list[nn.Parameter]
    objective: Callable
    n_train: int = 0
    batch_size: int = 0
    accuracy: Callable | None = None

    def batch(self, step, seed):
        if not self.batch_size:
            return None
        g = torch.Generator().manual_seed(seed * 100003 + step)
        return torch.randperm(self.n_train, generator=g)[:self.batch_size]


def make_problem(name, seed):
    if name not in DESCRIPTIONS:
        raise ValueError(f"Unknown problem {name}")
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    def param(value):
        return nn.Parameter(value.clone().to(torch.float64))

    def vector(value, fn):
        p = param(value.reshape(1, -1))
        return Problem([p], lambda split="train", batch=None: fn(p.flatten()))

    if name == "sphere":
        return vector(randn(12) * 2, lambda w: w.square().sum() / 2)
    if name == "ill_conditioned":
        a, _ = torch.linalg.qr(randn(64, 12))
        r, _ = torch.linalg.qr(randn(12, 12))
        a = 8 * a @ torch.diag(torch.logspace(0, -2, 12, dtype=torch.float64)) @ r.T
        y = a @ randn(12)
        return vector(torch.zeros(12), lambda w: (a @ w - y).square().mean() / 2)
    if name == "nonlinear_lse":
        x = torch.linspace(0, 2, 100, dtype=torch.float64)
        y = 1.7 * torch.exp(-1.2 * x) + 0.25
        return vector(torch.tensor([0.4, 0.8, 1.0]) + 0.05 * randn(3),
                      lambda w: (w[0] * torch.exp(w[1] * x) + w[2] - y).square().mean() / 2)
    if name == "rosenbrock":
        return vector(torch.tensor([-1.2, 1.0]) + 0.05 * randn(2),
                      lambda w: 100 * (w[1] - w[0] ** 2) ** 2 + (1 - w[0]) ** 2)
    if name == "beale":
        return vector(torch.tensor([-1.0, 1.0]) + 0.05 * randn(2),
                      lambda w: sum((c - w[0] + w[0] * w[1] ** k) ** 2
                                    for k, c in enumerate((1.5, 2.25, 2.625), 1)))
    if name == "rastrigin":
        return vector(2 * randn(8), lambda w: (w.square() + 10 * (1 - torch.cos(2 * torch.pi * w))).sum())
    if name == "matrix_factorization":
        target = randn(16, 3) @ randn(3, 12) / 3
        u, v = param(0.1 * randn(16, 3)), param(0.1 * randn(3, 12))
        return Problem([u, v], lambda split="train", batch=None: (u @ v - target).square().mean() / 2)

    # Independent train/validation/test examples generated from one common task.
    splits = {}
    if name in ("logistic", "noisy_regression"):
        truth = randn(12, 1) / 12 ** 0.5
        for split, n in (("train", 192), ("validation", 96), ("test", 192)):
            x = randn(n, 12) * torch.logspace(-0.6, 0.6, 12, dtype=torch.float64)
            if name == "logistic":
                y = (torch.rand(n, 1, generator=g) < torch.sigmoid(x @ truth)).to(torch.float64)
            else:
                y = x @ truth + 0.7 * randn(n, 1)
            splits[split] = (x, y)
        w = param(torch.zeros(12, 1))
        predict = lambda x: x @ w
        params = [w]
    else:
        for split, n in (("train", 192), ("validation", 96), ("test", 192)):
            if name == "xor":
                x = randn(n, 2)
                y = ((x[:, 0] * x[:, 1]) > 0).to(torch.float64).view(-1, 1)
            else:
                y = (torch.arange(n) % 2).to(torch.float64).view(-1, 1)
                angle = torch.rand(n, generator=g, dtype=torch.float64) * torch.pi
                x = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
                mask = y[:, 0] == 1
                x[mask, 0] = 1 - x[mask, 0]
                x[mask, 1] = 0.5 - x[mask, 1]
                x += 0.14 * randn(n, 2)
                if name == "noisy_labels" and split == "train":
                    flip = torch.rand(n, 1, generator=g) < 0.3
                    y = torch.where(flip, 1 - y, y)
            splits[split] = (x, y)
        w1, w2 = param(randn(3, 16) / 3 ** 0.5), param(randn(17, 1) / 17 ** 0.5)

        def predict(x):
            h = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), dim=1) @ w1
            h = torch.relu(h) if name == "xor" else torch.tanh(h)
            return torch.cat((h, torch.ones(len(h), 1, dtype=h.dtype)), dim=1) @ w2

        params = [w1, w2]

    def objective(split="train", batch=None):
        x, y = splits[split]
        if batch is not None:
            x, y = x[batch], y[batch]
        out = predict(x)
        if name == "noisy_regression":
            return (out - y).square().mean() / 2
        return F.binary_cross_entropy_with_logits(out, y)

    def accuracy(split):
        x, y = splits[split]
        return float(((predict(x) > 0) == y.bool()).to(torch.float64).mean())

    return Problem(params, objective, n_train=192,
                   batch_size=16 if name == "noisy_regression" else (32 if name == "noisy_labels" else 0),
                   accuracy=None if name == "noisy_regression" else accuracy)
