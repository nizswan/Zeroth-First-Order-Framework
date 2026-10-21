import copy
import math

import pytest
import torch

from ZFO import ZFO
from zfo import select_step


@pytest.mark.parametrize("method", ZFO.METHODS)
@pytest.mark.parametrize("inside", [True, False, "auto"])
def test_matches_research_selector(method, inside):
    p = torch.nn.Parameter(torch.tensor([1.2, -0.3], dtype=torch.float64))
    base = torch.optim.SGD([p], lr=0.2)
    adapter = ZFO(base, method=method, interior_search=inside, beta=2)
    original = p.detach().clone()
    delta = -0.2 * 2 * original
    phi = lambda a: -float(((original + a * delta) ** 2).sum())
    alpha, _ = select_step(phi, f0=phi(0), d1=-float((2 * original * delta).sum()),
                           dnorm=float(delta.norm()), method=method, bound=2,
                           epsilon=1e-3, search_inside=adapter.interior_search)
    result = adapter.step(lambda: p.square().sum())
    torch.testing.assert_close(p, original + alpha * delta)
    assert float(result) == float(original.square().sum())
    assert adapter.last_step["forwards"] == 3


def test_quadratic_interior_and_scheduler_groups():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    base = torch.optim.SGD([{"params": [p], "lr": 0.1}, {"params": [q], "lr": 0.2}])
    schedule = torch.optim.lr_scheduler.StepLR(base, step_size=1, gamma=0.5)
    adapter = ZFO(base, beta=10)
    adapter.step(lambda: p.square().sum() + q.square().sum())
    # Exact minimizer along the two-group displacement, not two separate lines.
    assert adapter.last_step["multiplier"] == pytest.approx(3.75, rel=1e-6)
    schedule.step()
    assert [g["lr"] for g in base.param_groups] == [0.05, 0.1]


def test_rejected_update_advances_momentum_once():
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    base = torch.optim.Adam([p], lr=5)
    adapter = ZFO(base, interior_search=False)
    adapter.step(lambda: p.square().sum())
    assert adapter.last_step["multiplier"] == 0
    assert p.item() == 1
    assert base.state[p]["step"].item() == 1
    assert base.state[p]["exp_avg"].item() == pytest.approx(0.2)


def test_probe_rng_and_buffers_match_one_training_forward():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.3))
    ref = copy.deepcopy(model)
    x = torch.randn(8, 4)
    rng = torch.get_rng_state()
    ref(x).square().mean().backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    adapter = ZFO(torch.optim.SGD(model.parameters(), lr=0.01), model=model)
    adapter.step(lambda: model(x).square().mean())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for a, b in zip(model.buffers(), ref.buffers()):
        torch.testing.assert_close(a, b)
    for a, b in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("error", ["nan", "raise"])
def test_probe_failure_rolls_back_all_state(error):
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2), torch.nn.Dropout())
    base = torch.optim.Adam(model.parameters())
    adapter = ZFO(base, model=model)
    x = torch.randn(5, 2)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    count = 0

    def closure():
        nonlocal count
        count += 1
        value = model(x).square().mean()
        if count == 3:
            if error == "raise":
                raise RuntimeError("probe error")
            value = value * float("nan")
        return value

    with pytest.raises((FloatingPointError, RuntimeError)):
        adapter.step(closure)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert not base.state
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(torch.get_rng_state(), rng)
    assert adapter.steps == 0


def test_checkpoint_resume():
    p = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    first = ZFO(torch.optim.Adam([p], lr=0.1), beta=2, method="taylor3")
    first.step(lambda: p.square().sum())
    q = torch.nn.Parameter(p.detach().clone())
    second = ZFO(torch.optim.Adam([q], lr=9))
    second.load_state_dict(first.state_dict())
    for _ in range(4):
        first.step(lambda: p.square().sum())
        second.step(lambda: q.square().sum())
    torch.testing.assert_close(p, q)
    assert second.last_step == first.last_step


def test_zero_direction_no_probes():
    p = torch.nn.Parameter(torch.tensor([0.0]))
    adapter = ZFO(torch.optim.SGD([p], lr=0.1))
    adapter.step(lambda: p.square().sum())
    assert adapter.last_step["forwards"] == 1
    assert adapter.last_step["multiplier"] == 0


def test_maximization_and_muon():
    p = torch.nn.Parameter(torch.ones(3, 2))
    adapter = ZFO(torch.optim.Muon([p], lr=0.02))
    adapter.step(lambda: p.square().mean())
    assert p.square().mean() < 1
    q = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    maximize = ZFO(torch.optim.SGD([q], lr=0.1, maximize=True), beta=10)
    maximize.step(lambda: -q.square().sum())
    assert abs(q.item()) < 1e-7


@pytest.mark.parametrize("kwargs", [{"epsilon": 0}, {"epsilon": float("nan")}, {"beta": -1},
                                    {"lr": float("inf")}, {"method": "unknown"}, {"interior_search": "yes"}])
def test_invalid_configuration(kwargs):
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError):
        ZFO(torch.optim.SGD([p], lr=0.1), **kwargs)


def test_weight_decay_is_part_of_displacement():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    base = torch.optim.AdamW([p], lr=0.1, weight_decay=0.2)
    ref = torch.optim.AdamW([q], lr=0.1, weight_decay=0.2)
    q.square().sum().backward()
    ref.step()
    adapter = ZFO(base, interior_search=False)
    adapter.step(lambda: p.square().sum())
    assert adapter.last_step["multiplier"] == 1
    torch.testing.assert_close(p, q)


@pytest.mark.parametrize("inside", ["auto", False, True])
def test_pade3_rational_interior_optimum_through_adapter(inside):
    # phi(a)=(1+a)/(1+a^2) has its maximum at sqrt(2)-1 in [0,2].
    # This must be an interior Padé choice, not a quadratic fallback.
    p = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    adapter = ZFO(torch.optim.SGD([p], lr=1, maximize=True),
                  method="pade3", beta=2, interior_search=inside)
    adapter.step(lambda: (1 + p) / (1 + p.square()))
    assert adapter.last_step["fallback"] == 0
    if inside is True:
        assert p.item() == pytest.approx(math.sqrt(2) - 1, abs=2e-6)
    else:
        assert p.item() == 0


@pytest.mark.parametrize("inside", ["auto", False, True])
def test_pade21_rational_optimum_and_probe_count(inside):
    # (1+2*x-x²)/(1+x) has a maximum at sqrt(2)-1.
    # Its derivative at zero is 1, so SGD with lr=1 proposes delta=1.
    p = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    adapter = ZFO(torch.optim.SGD([p], lr=1, maximize=True),
                  method="pade3_21", beta=2, interior_search=inside)
    calls = []

    def closure():
        calls.append(p.item())
        return (1 + 2*p - p.square()) / (1 + p)

    adapter.step(closure)
    assert len(calls) == 3
    assert adapter.last_step["fallback"] == 0
    expected = math.sqrt(2) - 1 if inside is True else 0.0
    assert p.item() == pytest.approx(expected, abs=2e-6)


def test_pade21_matches_series_and_handles_poles():
    from ZFO._search import _maximize_pade_21, select_model

    _, c, ok = _maximize_pade_21(2, 3, -4, 6, 1, True)
    assert ok
    # Multiply the desired Taylor series by the fitted denominator. The
    # cubic term must cancel, leaving the returned quadratic numerator.
    b = c["pade_b1"]
    assert c["pade_a0"] == 2
    assert c["pade_a1"] == pytest.approx(3 + 2*b)
    assert c["pade_a2"] == pytest.approx(-2 + 3*b)
    assert 1 - 2*b == pytest.approx(0)
    for d3 in (-12, -6, -6*(1-1e-10)):
        # b=-2, -1, and nearly -1: reject interior/endpoint/near poles.
        _, _, ok = _maximize_pade_21(1, 1, -2, d3, 1, True)
        assert not ok
        step, info = select_model("pade3_21", 1, 1, -2, d3, 1, 1.001, "true")
        assert info["fallback"] == 1
        assert step == pytest.approx(0.5)
    _, _, ok = _maximize_pade_21(1, 1, 0, 1, 1)
    assert not ok


def test_quadratic_radius_and_physical_probe_spacing():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    adapter = ZFO(torch.optim.SGD([p], lr=0.1), beta=10, epsilon=1e-3)
    probes = []

    def closure():
        probes.append(p.item())
        return p.square().sum()

    adapter.step(closure)
    assert probes == pytest.approx([2.0, 1.999, 2.001])
    assert p.item() == pytest.approx(0, abs=1e-7)
    assert adapter.last_step["multiplier"] == pytest.approx(5)
    assert adapter.last_step["radius"] == pytest.approx(4)


def test_legacy_default_checkpoint_resumes_with_current_selector():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    first = ZFO(torch.optim.Adam([p], lr=0.1), beta=2)
    first.step(lambda: p.square().sum())
    state = first.state_dict()
    state["zfo"]["search_type"] = "optimizer_displacement"
    state["zfo"]["last_step"]["search_type"] = "optimizer_displacement"
    q = torch.nn.Parameter(p.detach().clone())
    second = ZFO(torch.optim.Adam([q], lr=9))
    second.load_state_dict(state)
    assert "search_type" not in second.state_dict()["zfo"]
    assert second.last_step == first.last_step
    first.step(lambda: p.square().sum())
    second.step(lambda: q.square().sum())
    torch.testing.assert_close(p, q)
    assert second.last_step == first.last_step


@pytest.mark.parametrize("legacy_mode", ["normalized_direction", "unknown"])
def test_legacy_alternate_checkpoint_rejected_before_loading_optimizer(legacy_mode):
    p = torch.nn.Parameter(torch.ones(1))
    base = torch.optim.SGD([p], lr=0.1)
    adapter = ZFO(base)
    state = adapter.state_dict()
    state["zfo"]["search_type"] = legacy_mode
    state["optimizer"]["param_groups"][0]["lr"] = 10
    with pytest.raises(ValueError, match="search_type"):
        adapter.load_state_dict(state)
    assert base.param_groups[0]["lr"] == 0.1


@pytest.mark.parametrize("field,value", [("lr", -1), ("capturable", True)])
def test_invalid_checkpoint_restores_existing_state(field, value):
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    adapter = ZFO(torch.optim.Adam([p], lr=0.1), beta=2)
    adapter.step(lambda: p.square().sum())
    before = adapter.state_dict()
    invalid = copy.deepcopy(before)
    invalid["optimizer"]["param_groups"][0][field] = value
    invalid["optimizer"]["state"][0]["exp_avg"].fill_(999)
    invalid["zfo"]["beta"] = 7
    with pytest.raises(ValueError):
        adapter.load_state_dict(invalid)
    assert adapter.beta == 2
    assert adapter.steps == before["zfo"]["steps"]
    assert adapter.last_step == before["zfo"]["last_step"]
    assert adapter.param_groups[0]["lr"] == 0.1
    assert not adapter.param_groups[0]["capturable"]
    torch.testing.assert_close(adapter.optimizer.state[p]["exp_avg"],
                               before["optimizer"]["state"][0]["exp_avg"])


@pytest.mark.parametrize("method", ZFO.METHODS)
@pytest.mark.parametrize("base_name", ["SGD", "Adam", "AdamW", "Muon", "TACO"])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32), ("cpu", torch.float64),
    pytest.param("mps", torch.float32, marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS unavailable")),
    pytest.param("cuda", torch.float32, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_training_across_optimizers_and_selectors(method, base_name, device, dtype):
    from ZFO import TACO

    p = torch.nn.Parameter(torch.tensor([[2., -1.], [.5, 3.]], device=device, dtype=dtype))
    base_cls = TACO if base_name == "TACO" else getattr(torch.optim, base_name)
    adapter = ZFO(base_cls([p], lr=0.03), method=method,
                  beta=3, epsilon=0.05, interior_search=True)
    initial = p.square().mean().item()
    for _ in range(12):
        adapter.step(lambda: p.square().mean())
        assert 0 <= adapter.last_step["multiplier"] <= 3
        assert adapter.last_step["forwards"] <= 3
    assert p.square().mean().item() < initial


@pytest.mark.parametrize("method", ZFO.METHODS)
def test_supervised_classification_with_frozen_parameters(method):
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                torch.nn.Linear(4, 2)).double()
    model[0].requires_grad_(False)
    frozen = copy.deepcopy(model[0].state_dict())
    x = torch.randn(32, 3, dtype=torch.float64)
    y = (x[:, 0] > 0).long()
    adapter = ZFO(torch.optim.AdamW(model.parameters(), lr=0.05), model=model,
                  method=method, beta=3, interior_search=True)
    objective = lambda: torch.nn.functional.cross_entropy(model(x), y)
    initial = objective().item()
    for _ in range(20):
        adapter.step(objective)
    assert objective().item() < initial * 0.8
    for name, value in model[0].state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


def test_failed_probe_preserves_populated_state_and_next_update():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    adapter = ZFO(torch.optim.AdamW([p], lr=0.1), beta=2)
    adapter.step(lambda: p.square().sum())
    q = torch.nn.Parameter(p.detach().clone())
    reference = ZFO(torch.optim.AdamW([q]))
    reference.load_state_dict(adapter.state_dict())
    before_grad = p.grad.clone()
    before_info = copy.deepcopy(adapter.last_step)
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("failed second probe")
        return p.square().sum()

    with pytest.raises(RuntimeError, match="second probe"):
        adapter.step(closure)
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, before_grad, rtol=0, atol=0)
    assert adapter.last_step == before_info
    adapter.step(lambda: p.square().sum())
    reference.step(lambda: q.square().sum())
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert adapter.last_step == reference.last_step
