"""Run equal-grid scans with disjoint tuning/reporting seeds.

python -m benchmarks.suite --output benchmarks/results/reference --steps 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from ZFO import ZFO
from .tasks import DESCRIPTIONS, make_problem

BASES = ("SGD", "Adam", "Muon")
METHODS = ("flat", *ZFO.METHODS)
# Broad common numerical grid, independently selected for every pairing.
GRIDS = {base: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0] for base in BASES}


def optimizer_for(base, parameters, lr):
    if base == "SGD":
        return torch.optim.SGD(parameters, lr=lr, momentum=0, weight_decay=0)
    if base == "Adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=0)
    if base == "Muon":
        return torch.optim.Muon(parameters, lr=lr, momentum=0.95, nesterov=True,
                                ns_steps=5, weight_decay=0, adjust_lr_fn="match_rms_adamw")
    raise ValueError(base)


def run_trial(task, seed, base, method, lr, steps, beta, epsilon, interior_search, report=False):
    torch.manual_seed(seed)
    problem = make_problem(task, seed)
    base_opt = optimizer_for(base, problem.parameters, lr)
    opt = base_opt if method == "flat" else ZFO(
        base_opt, method=method, beta=beta, epsilon=epsilon, interior_search=interior_search)
    training_seconds = 0.0
    forwards = backwards = monitor_forwards = 0
    fallback = zero_steps = 0
    status, reason = "ok", ""
    history = []
    equal_work_test = equal_work_accuracy = math.nan
    equal_work_updates = 0
    budget = 3 * steps  # F + 2B; flat uses 3 units/update, ordinary ZFO uses 5.
    checkpoint_work = 0
    with torch.no_grad():
        initial = float(problem.objective())
    monitor_forwards += 1
    # In report runs, retain every checkpoint for plotting, never for LR tuning.
    if report:
        history.append(dict(update=0, train_loss=initial, work=0, seconds=0.0, multiplier=math.nan))
    for step in range(steps):
        batch = problem.batch(step, seed)
        calls = 0
        first_finite = False

        def closure():
            nonlocal forwards, calls, first_finite
            forwards += 1
            calls += 1
            value = problem.objective("train", batch)
            if calls == 1:
                first_finite = bool(torch.isfinite(value))
            return value
        start = time.perf_counter()
        try:
            if method == "flat":
                base_opt.zero_grad(set_to_none=True)
                loss = closure()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training objective")
                loss.backward()
                backwards += 1
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in problem.parameters):
                    raise FloatingPointError("Non-finite gradient")
                base_opt.step()
                alpha = 1.0
            else:
                opt.step(closure)
                backwards += 1
                alpha = opt.last_step["multiplier"]
                fallback += opt.last_step.get("fallback", 0)
                zero_steps += int(alpha == 0)
        except FloatingPointError as exc:
            # Algorithmic divergence is a result. Programming/runtime errors fail the suite.
            training_seconds += time.perf_counter() - start
            if method != "flat" and first_finite:
                backwards += 1
            status, reason = "diverged", str(exc)
            break
        training_seconds += time.perf_counter() - start
        with torch.no_grad():
            current = float(problem.objective())
        monitor_forwards += 1
        work = forwards + 2 * backwards
        if not math.isfinite(current) or current > 1e12:
            status, reason = "diverged", "Non-finite or >1e12 monitoring loss"
            break
        if report:
            history.append(dict(update=step + 1, train_loss=current, work=work,
                                seconds=training_seconds, multiplier=alpha))
            # Last checkpoint no later than the common work budget. Evaluation
            # does not feed back into optimization or hyperparameter selection.
            if work <= budget:
                with torch.no_grad():
                    equal_work_test = float(problem.objective("test"))
                    equal_work_accuracy = problem.accuracy("test") if problem.accuracy else math.nan
                monitor_forwards += 1 + int(problem.accuracy is not None)
                equal_work_updates, checkpoint_work = step + 1, work
    validation = test = accuracy = math.nan
    if status == "ok":
        with torch.no_grad():
            validation = float(problem.objective("validation"))
            if report:
                test = float(problem.objective("test"))
                accuracy = problem.accuracy("test") if problem.accuracy else math.nan
        monitor_forwards += 1 + int(report) * (1 + int(problem.accuracy is not None))
        if not math.isfinite(validation) or (report and not math.isfinite(test)):
            status, reason = "diverged", "Non-finite final evaluation"
    if status != "ok":
        validation = test = math.inf
        # A trajectory that fails after its compute checkpoint can still have a
        # valid equal-work result; report both statuses rather than erase it.
    row = dict(task=task, seed=seed, base=base, method=method, lr=lr,
               status=status, reason=reason, validation=validation, test=test,
               test_accuracy=accuracy, initial_loss=initial,
               train_seconds=training_seconds, forwards=forwards, backwards=backwards,
               work=forwards + 2 * backwards, monitor_forwards=monitor_forwards,
               equal_work_test=equal_work_test, equal_work_accuracy=equal_work_accuracy,
               equal_work_updates=equal_work_updates, checkpoint_work=checkpoint_work,
               zero_fraction=zero_steps / max(1, backwards), fallback_fraction=fallback / max(1, backwards))
    return row, history


def choose_rate(rows):
    frame = pd.DataFrame(rows)
    scores = []
    for lr, group in frame.groupby("lr", sort=True):
        # Any failed tuning seed makes that candidate ineligible; no nanmean.
        score = float(group.validation.mean()) if (group.status == "ok").all() else math.inf
        scores.append((score, float(lr)))
    score, rate = min(scores)
    return rate, math.isfinite(score)


def run_suite(output, *, tasks=None, steps=100, seeds=3, beta=3.0, epsilon=1e-3, interior_search="auto"):
    if seeds not in (3, 5) or steps < 1:
        raise ValueError("Use 3 or 5 seeds and at least one update")
    if beta <= 0 or epsilon <= 0 or not math.isfinite(beta + epsilon):
        raise ValueError("beta and epsilon must be finite and positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tasks = list(DESCRIPTIONS) if tasks is None else tasks
    if not set(tasks) <= set(DESCRIPTIONS):
        raise ValueError("Unknown task")
    tuning_seeds, report_seeds = list(range(100, 100 + seeds)), list(range(200, 200 + seeds))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Choose a fresh output directory; saved evidence is never silently overwritten")
    root = Path(__file__).resolve().parents[1]
    sources = ["zfo.py", "ZFO/_search.py", "ZFO/optimizer.py", "benchmarks/tasks.py", "benchmarks/suite.py"]
    manifest = dict(schema=1, status="running", tasks=tasks, steps=steps, tuning_seeds=tuning_seeds,
                    report_seeds=report_seeds, lr_grids=GRIDS, beta=beta, epsilon=epsilon,
                    interior_search=interior_search, dtype="float64", device="cpu", threads=1,
                    torch=torch.__version__, python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, platform=platform.platform(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in sources},
                    work_proxy="training forwards + 2 * backwards; excludes monitoring; not FLOPs",
                    seed_unit="problem/data realization + initialization + minibatch order",
                    muon="native torch Muon, every 2D tensor incl. augmented bias columns; rank-one matrices on vector tasks",
                    selection="lowest mean final validation across tuning seeds, all seeds must finish; ties choose smaller LR")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    scans, reports, curves, selections = [], [], [], []
    for task in tasks:
        for base in BASES:
            for method in METHODS:
                trials = []
                for lr in GRIDS[base]:
                    for seed in tuning_seeds:
                        row, _ = run_trial(task, seed, base, method, lr, steps, beta, epsilon, interior_search)
                        trials.append(row)
                        scans.append(row)
                lr, eligible = choose_rate(trials)
                selections.append(dict(task=task, base=base, method=method, lr=lr, tuning_valid=eligible,
                                       grid_edge=lr in (min(GRIDS[base]), max(GRIDS[base]))))
                for seed in report_seeds:
                    row, history = run_trial(task, seed, base, method, lr, steps, beta, epsilon, interior_search, report=True)
                    row["tuning_valid"] = eligible
                    reports.append(row)
                    curves.extend({"task": task, "base": base, "method": method, "seed": seed, **h} for h in history)
            print(f"{task}: {base} complete", flush=True)
        # Persist completed tasks while running; manifest status gates reporting.
        for filename, data in (("scans.csv", scans), ("runs.csv", reports), ("curves.csv", curves), ("selected.csv", selections)):
            pd.DataFrame(data).to_csv(output / filename, index=False)
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    tuning_runs=len(scans), report_runs=len(reports))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(DESCRIPTIONS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, choices=(3, 5), default=3)
    parser.add_argument("--beta", type=float, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--interior-search", choices=("auto", "true", "false"), default="auto")
    args = parser.parse_args()
    run_suite(args.output, tasks=args.tasks, steps=args.steps, seeds=args.seeds, beta=args.beta,
              epsilon=args.epsilon, interior_search=args.interior_search)


if __name__ == "__main__":
    main()
