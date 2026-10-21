import copy

import pytest
import torch

from ZFO import TACO, ZFO


def test_signed_column_top1_and_history():
    p = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.float64))
    optimizer = TACO([p], lr=0.2, momentum=0.5, spectral_scale=2)
    p.grad = torch.tensor([[1., -4.], [-3., 1.], [2., 2.]], dtype=p.dtype)
    optimizer.step()
    torch.testing.assert_close(p, torch.tensor([[0., .4], [.4, 0.], [0., 0.]], dtype=p.dtype))
    p.grad = torch.tensor([[0., 0.], [0., 0.], [8., -10.]], dtype=p.dtype)
    optimizer.step()
    torch.testing.assert_close(p, torch.tensor([[0., .4], [.4, 0.], [-.4, .4]], dtype=p.dtype))
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"],
                              torch.tensor([[.25, -1.], [-.75, .25], [4.5, -4.5]], dtype=p.dtype))


def test_ties_zero_and_missing_gradient_follow_reference():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    base = TACO([p], lr=1)
    p.grad = torch.tensor([[1., 0.], [-1., 0.]])
    base.step()
    torch.testing.assert_close(p, torch.tensor([[-1., 0.], [0., 0.]]))
    p.grad = None
    base.step()
    torch.testing.assert_close(p, torch.tensor([[-2., 0.], [0., 0.]]))


def test_taco_zfo_resume_and_scheduler():
    p = torch.nn.Parameter(torch.tensor([[2., -1.], [.5, 3.]], dtype=torch.float64))
    base = TACO([p], lr=.2)
    adapter = ZFO(base, method="pade3_21", interior_search=True, beta=3)
    adapter.step(lambda: p.square().sum())
    q = torch.nn.Parameter(p.detach().clone())
    other = ZFO(TACO([q]))
    other.load_state_dict(copy.deepcopy(adapter.state_dict()))
    for _ in range(3):
        adapter.step(lambda: p.square().sum())
        other.step(lambda: q.square().sum())
    torch.testing.assert_close(p, q)
    assert adapter.last_step == other.last_step
    scheduler = torch.optim.lr_scheduler.StepLR(base, step_size=1, gamma=.5)
    adapter.step(lambda: p.square().sum())
    scheduler.step()
    assert base.param_groups[0]["lr"] == .1


@pytest.mark.parametrize("kwargs", [{"lr": -1}, {"momentum": 1}, {"spectral_scale": float("nan")}])
def test_invalid_taco_configuration(kwargs):
    with pytest.raises(ValueError):
        TACO([torch.nn.Parameter(torch.ones(2, 2))], **kwargs)


def test_nonmatrix_rejected():
    with pytest.raises(ValueError, match="2D"):
        TACO([torch.nn.Parameter(torch.ones(2))])
