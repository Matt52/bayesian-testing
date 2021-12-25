import numpy as np
import pytest
from bayes_ab_test.metrics.simulations import beta_posteriors_all, lognormal_posteriors

BETA_POSTERIORS_ALL_INPUTS = [
    {
        "totals": [10, 20, 30],
        "successes": [8, 16, 24],
        "sim_count": 10,
        "a_priors_beta": [0.5, 0.5, 0.5],
        "b_priors_beta": [0.5, 0.5, 0.5],
    },
    {
        "totals": [20, 30],
        "successes": [16, 24],
        "sim_count": 20,
        "a_priors_beta": [0.5, 0.5],
        "b_priors_beta": [0.5, 0.5],
    },
]

LOGNORMAL_POSTERIORS_INPUTS = [
    {
        "totals": 1580,
        "sum_logs": 3831.806394737816,
        "sum_logs_2": 11029.923165846496,
        "sim_count": 10000,
    },
    {
        "totals": 1580,
        "sum_logs": 4055.965234848171,
        "sum_logs_2": 12357.911862914,
        "sim_count": 100,
    },
    {
        "totals": 0,
        "sum_logs": 0,
        "sum_logs_2": 0,
        "sim_count": 100,
    },
]


@pytest.mark.parametrize("inp", BETA_POSTERIORS_ALL_INPUTS)
def test_beta_posteriors_all(inp):
    all_pos = beta_posteriors_all(
        inp["totals"],
        inp["successes"],
        inp["sim_count"],
        inp["a_priors_beta"],
        inp["b_priors_beta"],
    )
    all_pos_shape = np.array(all_pos).shape
    assert all_pos_shape == (len(inp["totals"]), inp["sim_count"])


@pytest.mark.parametrize("inp", LOGNORMAL_POSTERIORS_INPUTS)
def test_lognormal_posteriors(inp):
    all_pos = lognormal_posteriors(
        inp["totals"],
        inp["sum_logs"],
        inp["sum_logs_2"],
        inp["sim_count"],
    )
    assert len(all_pos) == inp["sim_count"]
