"""Matrix-only, full-precision TACO adapter for local-model experiments.

Reference: https://anonymous.4open.science/r/iclr27-TACO-923D/
Test_TACO_FO_v8.2.ipynb: normalized EMA, l1l1_op_norm (k=1), and
apply_sparse_taco_update_. This implements the dense-history rule without
backward hooks, quantization, compressed history, or architecture routing.
It is not a reproduction of the notebook's E4M3/hk16 experiment configuration.
"""

import math

import torch
from torch.optim import Optimizer


class TACO(Optimizer):
    """Column-wise signed top-1 updates from an exponential gradient average.

    Parameters must be logical 2D matrices in float32/float64. Each column
    updates the row with largest absolute momentum (first row wins ties).
    ``momentum`` is the EMA decay; ``spectral_scale`` is an explicit per-group
    factor (default 1). Set it to fan_out/fan_in for the reference FFN policy.
    Unobserved matrices decay existing history and retain its signed update,
    matching the reference dense, non-conditional matrix path.
    """

    def __init__(self, params, lr=3e-5, momentum=0.95, spectral_scale=1.0,
                 maximize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                    spectral_scale=spectral_scale, maximize=maximize))
        self._validate()

    def _validate(self):
        for group in self.param_groups:
            for name in ("lr", "spectral_scale", "momentum"):
                value = float(group[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
            if group["momentum"] >= 1:
                raise ValueError("momentum must be less than 1")
            for p in group["params"]:
                if p.ndim != 2 or 0 in p.shape:
                    raise ValueError("TACO requires nonempty logical 2D matrices")
                if p.dtype not in (torch.float32, torch.float64):
                    raise TypeError("This TACO adapter requires float32/float64 parameters")
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise TypeError("TACO requires dense gradients")
                    if not bool(torch.isfinite(p.grad).all()):
                        raise FloatingPointError("Non-finite TACO gradient")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._validate()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None and not self.state.get(p):
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                history = state["momentum_buffer"]
                history.mul_(group["momentum"])
                if p.grad is not None:
                    history.add_(p.grad, alpha=1 - group["momentum"])
                rows = history.abs().argmax(dim=0)
                columns = torch.arange(p.shape[1], device=p.device)
                sign = 1 if group["maximize"] else -1
                p[rows, columns] += (sign * group["lr"] * group["spectral_scale"]
                                     * history[rows, columns].sign())
        return loss
