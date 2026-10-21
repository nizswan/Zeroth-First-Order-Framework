"""Fixed-budget Taylor/Padé search and parameter-space operations."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn


def _trainable_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _snapshot_trainable(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in _trainable_params(model)}


def _snapshot_trainable_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for n, p in _trainable_params(model):
        if p.grad is None:
            out[n] = torch.zeros_like(p)
        else:
            out[n] = p.grad.detach().clone()
    return out


@torch.no_grad()
def _apply_trainable(model: nn.Module, snap: Dict[str, torch.Tensor]) -> None:
    for n, p in _trainable_params(model):
        p.copy_(snap[n])


@torch.no_grad()
def _apply_trainable_blend(
    model: nn.Module,
    alpha: float,
    theta: Dict[str, torch.Tensor],
    theta_hat: Dict[str, torch.Tensor],
) -> None:
    a = float(alpha)
    for n, p in _trainable_params(model):
        p.copy_(theta[n] + a * (theta_hat[n] - theta[n]))


def _delta(
    theta: Dict[str, torch.Tensor], theta_hat: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {k: theta_hat[k] - theta[k] for k in theta.keys()}


def _dot(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = None
    for k in a.keys():
        v = (a[k] * b[k]).sum()
        s = v if s is None else s + v
    if s is None:
        return torch.zeros((), device=next(iter(a.values())).device)
    return s


def _norm(a: Dict[str, torch.Tensor]) -> float:
    return float(torch.sqrt(_dot(a, a).clamp(min=0.0)).item())


# Shared with the installable adapter; preserve the legacy import surface.
from ZFO._search import (
    _maximize_quadratic,
    _maximize_cubic,
    _maximize_pade_11,
    _pade12_coeffs,
    _pade12_has_pole_in_interval,
    _maximize_pade_12,
    _real_roots,
    select_model,
    select_step,
)
