import math
import subprocess
import sys

import pytest
import torch
import pandas as pd

from benchmarks.report import equal_work_losses, mean_std
from benchmarks.suite import BASES, GRIDS, METHODS, choose_rate, run_trial
from benchmarks.tasks import DESCRIPTIONS, make_problem


def test_flat_refinement_preserves_original_candidates():
    original = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    assert METHODS == ("flat", "taylor2", "taylor3", "pade2", "pade3")
    for base in BASES:
        flat = GRIDS[base]["flat"]
        assert len(flat) == 13
        assert flat[::2] == original
        assert all(b / a == pytest.approx(math.sqrt(10)) for a, b in zip(flat, flat[1:]))
        for method in METHODS[1:]:
            assert GRIDS[base][method] == original


def test_bad_seed_is_not_dropped_from_rate_selection():
    rows = [dict(lr=0.1, validation=0.001, status="ok"),
            dict(lr=0.1, validation=math.inf, status="diverged"),
            dict(lr=0.01, validation=0.2, status="ok"),
            dict(lr=0.01, validation=0.3, status="ok")]
    assert choose_rate(rows) == (0.01, True)
    assert choose_rate([dict(lr=1, validation=math.inf, status="diverged")]) == (1, False)


def test_equal_work_rejects_failure_before_budget_but_keeps_later_checkpoint():
    runs = pd.DataFrame([
        dict(status="ok", work=300, equal_work_test=0.2),
        dict(status="diverged", work=30, equal_work_test=0.3),
        dict(status="diverged", work=300, equal_work_test=0.4),
        dict(status="diverged", work=305, equal_work_test=0.5),
    ])
    losses = equal_work_losses(runs, {"steps": 100})
    assert losses.iloc[0] == 0.2
    assert math.isnan(losses.iloc[1]) and math.isnan(losses.iloc[2])
    assert losses.iloc[3] == 0.5
    assert mean_std(losses) == "failed / incomplete"


@pytest.mark.parametrize("task", list(DESCRIPTIONS))
def test_problem_directional_derivative(task):
    problem = make_problem(task, 200)
    f = problem.objective()
    grads = torch.autograd.grad(f, problem.parameters)
    generator = torch.Generator().manual_seed(99)
    directions = [torch.randn(p.shape, dtype=p.dtype, generator=generator) for p in problem.parameters]
    expected = sum(float((g * d).sum()) for g, d in zip(grads, directions))
    originals = [p.detach().clone() for p in problem.parameters]
    values = []
    with torch.no_grad():
        for scale in (1e-6, -1e-6):
            for p, original, d in zip(problem.parameters, originals, directions):
                p.copy_(original + scale * d)
            values.append(float(problem.objective()))
        for p, original in zip(problem.parameters, originals):
            p.copy_(original)
    assert (values[0] - values[1]) / 2e-6 == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_measured_work_and_reproducible_batches():
    torch.set_num_threads(1)
    args = ("noisy_regression", 200, "Adam", "taylor2", 0.01, 5, 3, 1e-3, "auto")
    first, curves = run_trial(*args, report=True)
    second, _ = run_trial(*args, report=True)
    assert first["test"] == second["test"]
    assert first["forwards"] == 15
    assert first["backwards"] == 5
    assert first["work"] == 25
    assert first["equal_work_updates"] == 3
    assert curves[-1]["work"] == 25


def test_installed_import_outside_checkout(tmp_path):
    # Catches case-insensitive filesystems resolving `zfo` to the `ZFO` package.
    subprocess.run([sys.executable, "-c", "from ZFO import ZFO; import torch; p=torch.nn.Parameter(torch.ones(1)); o=ZFO(torch.optim.SGD([p],lr=.1)); o.step(lambda:p.square().sum())"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
