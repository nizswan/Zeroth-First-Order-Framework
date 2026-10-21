"""A closure-based adapter around a conventional PyTorch optimizer.

The base optimizer proposes a displacement once. The existing research selector
chooses its multiplier using two fixed-batch forward probes. Optimizer moments
advance once even when that multiplier is zero, as in the research code.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer

from ._search import select_step


class ZFO:
    """Wrap an optimizer without changing the model or data loader.

    ``closure`` returns a scalar differentiable objective; it must NOT call
    backward, zero_grad, or advance a data iterator. The adapter owns backward.
    Use the same batch and objective for every closure call in a step. Supply
    ``model`` to restore mutable buffers (e.g. BatchNorm) around probe forwards.
    Torch CPU/device RNG is replayed so dropout uses the same masks.

    Attach ordinary LR schedulers to ``optimizer`` and call the scheduler after
    this adapter's step. ``lr=None`` preserves every group's existing rate.
    This is deliberately an adapter, not a torch.optim.Optimizer subclass.
    """

    METHODS = ("taylor2", "taylor3", "pade2", "pade3")

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        epsilon: float = 1e-3,
        method: str = "taylor2",
        interior_search: bool | str = "auto",
        beta: float = 1.0,
        lr: float | None = None,
        model: nn.Module | None = None,
    ):
        if not isinstance(optimizer, Optimizer) or isinstance(optimizer, torch.optim.LBFGS):
            raise TypeError("ZFO requires a closure-free torch optimizer (e.g. SGD, Adam, Muon)")
        self.optimizer = optimizer
        self.model = model
        self.epsilon = self._positive("epsilon", epsilon)
        self.beta = self._positive("beta", beta)
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.method = method
        self.interior_search = self._inside(interior_search)
        if lr is not None:
            rate = self._positive("lr", lr, zero=True)
            for group in optimizer.param_groups:
                group["lr"] = rate
        self.steps = 0
        self.last_step: dict = {}
        self._params()  # Fail early on unsupported parameter layouts.

    @staticmethod
    def _positive(name, value, zero=False):
        value = float(value)
        if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
            raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
        return value

    @staticmethod
    def _inside(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if value not in ("auto", "true", "false"):
            raise ValueError("interior_search must be a bool or auto/true/false")
        return value

    def _params(self):
        params = [p for g in self.param_groups for p in g["params"]]
        if not params or len({id(p) for p in params}) != len(params):
            raise ValueError("ZFO needs nonempty, nonduplicated optimizer parameters")
        if any(p.dtype not in (torch.float32, torch.float64) for p in params):
            raise TypeError("Use float32/float64 parameters; low-precision probe storage is unsupported")
        if len({p.device for p in params}) != 1:
            raise ValueError("This adapter supports one parameter device per optimizer")
        if params[0].device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("Supported parameter devices: CPU, CUDA, MPS")
        if any(g.get("differentiable", False) or g.get("capturable", False) for g in self.param_groups):
            raise ValueError("Differentiable/capturable optimizer steps are unsupported")
        if len({bool(g.get("maximize", False)) for g in self.param_groups}) != 1:
            raise ValueError("All parameter groups must agree on maximize")
        for group in self.param_groups:
            self._positive("group lr", group["lr"], zero=True)
        return params

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _rng(device):
        state = {"cpu": torch.get_rng_state().clone()}
        if device.type == "cuda":
            state["device"] = torch.cuda.get_rng_state(device).clone()
        elif device.type == "mps":
            state["device"] = torch.mps.get_rng_state().clone()
        return state

    @staticmethod
    def _set_rng(state, device):
        torch.set_rng_state(state["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["device"], device)
        elif device.type == "mps":
            torch.mps.set_rng_state(state["device"])

    @staticmethod
    def _copy(destination, source):
        with torch.no_grad():
            for dst, src in zip(destination, source):
                dst.copy_(src)

    @staticmethod
    def _loss(value):
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("closure must return one scalar Tensor")
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Non-finite closure objective")
        return value

    def step(self, closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Take one selected step; return the detached objective before updating.

        One backward and (normally) three forwards: original, +epsilon, -epsilon.
        No extra acceptance forward is hidden here: a model-selected step need
        not improve the actual objective. Inspect ``last_step`` for diagnostics.
        Exceptions restore parameters, gradients, optimizer state, buffers, RNG.
        """
        if not callable(closure):
            raise TypeError("step requires a fixed-batch, forward-only closure")
        params = self._params()
        device = params[0].device
        theta = [p.detach().clone() for p in params]
        old_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        buffers = list(self.model.buffers()) if self.model is not None else []
        before_buffers = [b.detach().clone() for b in buffers]
        before_rng = self._rng(device)
        try:
            self.zero_grad()
            with torch.enable_grad():
                loss = self._loss(closure())
                loss.backward()
            if any(p.grad is not None and p.grad.is_sparse for p in params):
                raise TypeError("Sparse gradients are unsupported")
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError("Non-finite gradient")
            grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
            after_buffers = [b.detach().clone() for b in buffers]
            after_rng = self._rng(device)
            self.optimizer.step()
            delta = [p.detach() - t for p, t in zip(params, theta)]
            if any(not bool(torch.isfinite(d).all()) for d in delta):
                raise FloatingPointError("Non-finite optimizer displacement")
            # Accumulate in float64 on the host; probes still use parameter dtype.
            norm = math.sqrt(sum(float(d.double().square().sum().item()) if device.type != "mps"
                                 else float(d.float().square().sum().item()) for d in delta))
            dot = sum(float((g.double() * d.double()).sum().item()) if device.type != "mps"
                      else float((g.float() * d.float()).sum().item())
                      for g, d in zip(grads, delta) if g is not None)
            sign = 1.0 if self.param_groups[0].get("maximize", False) else -1.0

            def place(alpha):
                with torch.no_grad():
                    for p, t, d in zip(params, theta, delta):
                        p.copy_(t + alpha * d)

            @torch.no_grad()
            def phi(alpha):
                place(alpha)
                self._copy(buffers, before_buffers)
                self._set_rng(before_rng, device)
                return sign * float(self._loss(closure()).item())

            alpha, info = select_step(
                phi, f0=sign * float(loss.detach().item()), d1=sign * dot,
                dnorm=norm, lr=float(self.param_groups[0]["lr"]),
                method=self.method, bound=self.beta, epsilon=self.epsilon,
                search_inside=self.interior_search,
            )
            place(alpha)
            self._copy(buffers, after_buffers)
            self._set_rng(after_rng, device)
            self.steps += 1
            self.last_step = {
                **info, "step": self.steps, "method": self.method,
                "multiplier": alpha, "loss_before": float(loss.detach().item()),
                "forwards": 1 + info["n_extra_forwards"], "backwards": 1,
                "effective_lrs": [alpha * float(g["lr"]) for g in self.param_groups],
            }
            return loss.detach()
        except Exception:
            self._copy(params, theta)
            self.optimizer.load_state_dict(saved_optimizer)
            for p, grad in zip(params, old_grads):
                p.grad = grad
            self._copy(buffers, before_buffers)
            self._set_rng(before_rng, device)
            raise

    def state_dict(self):
        """Optimizer + ZFO settings; save model/scheduler/RNG states separately."""
        return copy.deepcopy({
            "version": 1, "optimizer": self.optimizer.state_dict(),
            "zfo": {"epsilon": self.epsilon, "method": self.method,
                    "interior_search": self.interior_search, "beta": self.beta,
                    "steps": self.steps, "last_step": self.last_step},
        })

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported ZFO checkpoint version")
        config = state["zfo"]
        epsilon = self._positive("epsilon", config["epsilon"])
        beta = self._positive("beta", config["beta"])
        inside = self._inside(config["interior_search"])
        if config["method"] not in self.METHODS:
            raise ValueError("Unknown checkpoint method")
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.epsilon, self.beta, self.interior_search = epsilon, beta, inside
        self.method, self.steps = config["method"], config["steps"]
        self.last_step = copy.deepcopy(config["last_step"])
