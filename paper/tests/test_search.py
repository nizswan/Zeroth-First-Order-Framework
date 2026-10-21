import pytest

import baselines
import zfo
from ZFO._search import select_step as library_step


@pytest.mark.parametrize("method", ["taylor2", "taylor3", "pade2", "pade3"])
def test_paper_and_library_share_local_models(method):
    objective = lambda a: -(a - 0.4) ** 2
    settings = dict(f0=objective(0), d1=0.8, dnorm=2, method=method,
                    bound=1, epsilon=0.01)
    assert zfo.select_step(objective, **settings) == library_step(objective, **settings)


def test_sequential_baseline_remains_available_to_paper_runner():
    objective = lambda a: -(a - 0.4) ** 2
    expected = baselines.select_zfo_seq(
        objective, phi0=objective(0), d1=0.8, dalpha=0.005,
        beta=1, k=5, initial_model="p3", search_inside="auto",
    )
    alpha, info = zfo.select_step(
        objective, f0=objective(0), d1=0.8, dnorm=2, method="zfo_seq",
        bound=1, epsilon=0.01,
    )
    assert alpha == expected.selected_a
    assert info["n_extra_forwards"] == expected.probes_used
    assert info["radius"] == 2
