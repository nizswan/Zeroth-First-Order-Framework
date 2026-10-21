"""Shared research Taylor/Pade selectors; formulas and safeguards preserved."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _maximize_quadratic(
    f0: float, d1: float, d2: float, bound: float, search_inside: bool = True
) -> float:
    cand = [0.0, float(bound)]
    if search_inside and abs(d2) > 1e-12:
        a_star = -d1 / d2
        if 0.0 <= a_star <= float(bound):
            cand.append(float(a_star))

    def q(a: float) -> float:
        return f0 + a * d1 + 0.5 * a * a * d2

    best_a, best_v = (cand[0], -1e30)
    for a in cand:
        v = q(a)
        if v > best_v:
            best_v, best_a = (v, a)
    return float(best_a)


def _maximize_cubic(
    f0: float, d1: float, d2: float, d3: float, bound: float, search_inside: bool = True
) -> float:
    cand = [0.0, float(bound)]
    A = 0.5 * d3
    B = d2
    C = d1
    if search_inside:
        if abs(A) < 1e-12:
            if abs(B) > 1e-12:
                r = -C / B
                if 0.0 <= r <= float(bound):
                    cand.append(float(r))
        else:
            disc = B * B - 4.0 * A * C
            if disc >= 0.0:
                sd = math.sqrt(disc)
                r1 = (-B + sd) / (2.0 * A)
                r2 = (-B - sd) / (2.0 * A)
                if 0.0 <= r1 <= float(bound):
                    cand.append(float(r1))
                if 0.0 <= r2 <= float(bound):
                    cand.append(float(r2))

    def q(a: float) -> float:
        return f0 + a * d1 + 0.5 * a * a * d2 + 1.0 / 6.0 * a * a * a * d3

    best_a, best_v = (cand[0], -1e30)
    for a in cand:
        v = q(a)
        if v > best_v:
            best_v, best_a = (v, a)
    return float(best_a)


def _maximize_pade_11(
    f0: float, d1: float, d2: float, bound: float
) -> Tuple[float, Dict[str, float], bool]:
    info: Dict[str, float] = {}
    if abs(d1) < 1e-12:
        return (float(bound), info, True)
    a0 = f0
    b1 = -d2 / (2.0 * d1)
    a1 = d1 + a0 * b1
    info.update({"pade_a0": float(a0), "pade_a1": float(a1), "pade_b1": float(b1)})

    def q(a: float) -> float:
        den = 1.0 + b1 * a
        if abs(den) < 1e-09:
            return -1e30
        return (a0 + a1 * a) / den

    v0 = q(0.0)
    vb = q(float(bound))
    alpha = float(bound) if vb >= v0 else 0.0
    ok = True
    return (alpha, info, ok)


def _pade12_coeffs(
    f0: float, d1: float, d2: float, d3: float
) -> Tuple[Dict[str, float], bool]:
    a0 = f0
    denom = d1 * d1 - 0.5 * a0 * d2
    if abs(denom) < 1e-12:
        return ({}, False)
    b1 = (a0 * (d3 / 6.0) - 0.5 * d1 * d2) / denom
    if abs(a0) < 1e-12:
        return ({}, False)
    b2 = (-0.5 * d2 - d1 * b1) / a0
    a1 = d1 + a0 * b1
    return (
        {
            "pade_a0": float(a0),
            "pade_a1": float(a1),
            "pade_b1": float(b1),
            "pade_b2": float(b2),
        },
        True,
    )


def _pade12_has_pole_in_interval(b1: float, b2: float, bound: float) -> bool:
    if abs(b2) < 1e-14:
        if abs(b1) < 1e-14:
            return False
        r = -1.0 / b1
        return 0.0 <= r <= float(bound)
    disc = b1 * b1 - 4.0 * b2
    if disc < 0.0:
        return False
    sd = math.sqrt(disc)
    r1 = (-b1 + sd) / (2.0 * b2)
    r2 = (-b1 - sd) / (2.0 * b2)
    return 0.0 <= r1 <= float(bound) or 0.0 <= r2 <= float(bound)


def _maximize_pade_12(
    f0: float,
    d1: float,
    d2: float,
    d3: float,
    bound: float,
    search_inside: bool = False,
) -> Tuple[float, Dict[str, float], bool]:
    coeffs, ok = _pade12_coeffs(f0=f0, d1=d1, d2=d2, d3=d3)
    if not ok:
        return (0.0, {}, False)
    a0 = coeffs["pade_a0"]
    a1 = coeffs["pade_a1"]
    b1 = coeffs["pade_b1"]
    b2 = coeffs["pade_b2"]
    if _pade12_has_pole_in_interval(float(b1), float(b2), float(bound)):
        return (0.0, coeffs, False)

    def q(a: float) -> float:
        den = 1.0 + b1 * a + b2 * a * a
        if abs(den) < 1e-09:
            return -1e30
        return (a0 + a1 * a) / den

    denb = 1.0 + b1 * float(bound) + b2 * float(bound) * float(bound)
    if abs(denb) < 1e-09:
        return (0.0, coeffs, False)
    v0 = q(0.0)
    vb = q(float(bound))
    alpha = float(bound) if vb >= v0 else 0.0
    for t in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        a = float(bound) * float(t)
        den = 1.0 + b1 * a + b2 * a * a
        if abs(den) < 1e-06:
            return (0.0, coeffs, False)
    if search_inside:
        # Numerator of P12': (a1-a0*b1) - 2*a0*b2*x - a1*b2*x*x.
        for root in _real_roots(-a1 * b2, -2.0 * a0 * b2, a1 - a0 * b1):
            if 0.0 < root < bound and q(root) > q(alpha):
                alpha = root
    return (float(alpha), coeffs, True)


def _maximize_pade_21(
    f0: float, d1: float, d2: float, d3: float, bound: float,
    search_inside: bool = False,
) -> Tuple[float, Dict[str, float], bool]:
    """Match f0 + d1*x + d2*x²/2 + d3*x³/6 with a [2/1] model.

    This library extension uses the same local information as [1/2]. Reject
    nonfinite coefficients and any pole in the closed search interval.
    """
    if not all(math.isfinite(v) for v in (f0, d1, d2, d3, bound)) or abs(d2) < 1e-14:
        return 0.0, {}, False
    c2 = d2 / 2.0
    b1 = -d3 / (3.0 * d2)
    a0, a1, a2 = f0, d1 + f0 * b1, c2 + d1 * b1
    coeffs = dict(pade_a0=a0, pade_a1=a1, pade_a2=a2, pade_b1=b1)
    if not all(math.isfinite(v) for v in coeffs.values()):
        return 0.0, {}, False
    # The linear denominator starts at 1, so a nonpositive endpoint implies
    # a pole; also reject near-zero endpoints before evaluating candidates.
    denb = 1.0 + b1 * bound
    if not math.isfinite(denb) or denb < 1e-9:
        return 0.0, coeffs, False

    def q(x):
        # Equivalent to (a0+a1*x+a2*x²)/(1+b1*x), with less cancellation.
        return f0 + d1 * x + c2 * x * x / (1.0 + b1 * x)

    candidates = [0.0, float(bound)]
    if search_inside:
        # Numerator of the derivative: d1 + 2*a2*x + a2*b1*x².
        candidates.extend(r for r in _real_roots(a2 * b1, 2.0 * a2, d1)
                          if 0.0 < r < bound)
    values = [q(x) for x in candidates]
    if not all(math.isfinite(v) for v in values):
        return 0.0, coeffs, False
    # Match the existing rational selector's endpoint tie convention.
    best = 1 if values[1] >= values[0] else 0
    for i in range(2, len(candidates)):
        if values[i] > values[best]:
            best = i
    return float(candidates[best]), coeffs, True


def _real_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots, using a stable quadratic formula and relative scaling."""
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0 or not math.isfinite(scale):
        return []
    a, b, c = a / scale, b / scale, c / scale
    if abs(a) < 1e-14:
        return [] if abs(b) < 1e-14 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
    return [-b / (2.0 * a)] if q == 0 else [q / a, c / q]


def select_model(method, f0, d1, d2, d3, bound, fp, search_inside="auto"):
    """Maximize a local model, preserving the experiment's default safeguards.

    auto: Taylor stationary points; Padé endpoints; quadratic fallbacks may be
    interior. false: endpoints only, including fallbacks. true: also include
    stationary points of Taylor2, Taylor3 and pole-free Padé3.
    """
    if search_inside not in {"auto", "true", "false"}:
        raise ValueError("search_inside must be auto, true, or false")
    inside = search_inside != "false"
    fallback = 0
    reason = 0
    info = {}

    def quadratic():
        return _maximize_quadratic(f0, d1, d2, bound, search_inside=inside)

    if abs(d2) < 1e-14:
        step = float(bound) if fp >= f0 else 0.0
        fallback, reason = 1, 1
    elif method == "taylor2":
        step = quadratic()
    elif method == "taylor3":
        if d3 is None or not math.isfinite(d3):
            step, fallback, reason = quadratic(), 1, 2
        elif abs(d3) > 1e6 * max(1.0, abs(d2)):
            step, fallback, reason = quadratic(), 1, 3
        else:
            step = _maximize_cubic(f0, d1, d2, d3, bound, search_inside=inside)
    elif method == "pade2":
        step, info, ok = _maximize_pade_11(f0, d1, d2, bound)
        if not ok:
            step, fallback, reason = quadratic(), 1, 4
    elif method in {"pade3", "pade3_21"}:
        if d3 is None or not math.isfinite(d3):
            step, fallback, reason = quadratic(), 1, 5
        else:
            maximize = _maximize_pade_12 if method == "pade3" else _maximize_pade_21
            step, info, ok = maximize(
                f0, d1, d2, d3, bound, search_inside=search_inside == "true"
            )
            if not ok:
                step, fallback, reason = quadratic(), 1, 6
    else:
        raise ValueError(f"Unknown local model: {method}")
    if not math.isfinite(step):
        raise FloatingPointError("Non-finite search step")
    return max(0.0, min(float(bound), float(step))), {
        "fallback": fallback,
        "fallback_reason": reason,
        **info,
    }


def select_step(
    phi,
    *,
    f0,
    d1,
    dnorm,
    method,
    bound,
    epsilon,
    search_inside="auto",
):
    """Return a multiplier of the realized optimizer displacement.

    phi(a) evaluates parameters theta + a*d for a in [0, bound]. The
    physical radius is bound*||d||; symmetric probes have distance epsilon.
    All local models and safeguards use displacement-multiplier coordinates.
    """
    if not math.isfinite(dnorm) or not math.isfinite(d1):
        raise FloatingPointError("Non-finite optimizer direction or derivative")
    if dnorm < 1e-20:
        return 0.0, {"dnorm": dnorm, "n_extra_forwards": 0, "fallback": 1}
    radius = bound * dnorm
    probe_spacing = epsilon / dnorm
    fp, fm = phi(probe_spacing), phi(-probe_spacing)
    if not all(math.isfinite(v) for v in (f0, fp, fm)):
        raise FloatingPointError("Non-finite fixed-batch probe")
    # Preserve the order of operations used in the large-model experiments.
    phi2 = (fp - 2.0 * f0 + fm) / epsilon**2
    d2 = dnorm**2 * phi2
    d3 = None
    if method in {"taylor3", "pade3", "pade3_21"}:
        phi3 = 3.0 * ((fp - fm) - 2.0 * probe_spacing * d1) / epsilon**3
        d3 = dnorm**3 * phi3
    step, info = select_model(method, f0, d1, d2, d3, bound, fp, search_inside)
    return step, {
        "f_theta": f0,
        "f_p": fp,
        "f_m": fm,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "dnorm": dnorm,
        "dalpha": epsilon / dnorm,
        "radius": radius,
        "n_extra_forwards": 2,
        **info,
    }
