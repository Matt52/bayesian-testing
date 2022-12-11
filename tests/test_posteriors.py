import numpy as np
import pytest

from bayesian_testing.metrics.posteriors import (
    beta_posteriors_all,
    lognormal_posteriors,
    dirichlet_posteriors,
    gamma_posteriors_all,
)

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

DIRICHLET_POSTERIORS_INPUTS = [
    {
        "concentration": [1, 2, 3],
        "prior": [1, 1, 1],
        "sim_count": 10000,
    },
    {
        "concentration": [100, 200],
        "prior": [1 / 2, 1 / 2],
        "sim_count": 100,
    },
]

GAMMA_POSTERIORS_ALL_INPUTS = [
    {
        "totals": [10, 20, 30],
        "sums": [80, 161, 260],
        "sim_count": 10,
        "a_priors_gamma": [0.5, 0.5, 0.5],
        "b_priors_gamma": [0.5, 0.5, 0.5],
    },
    {
        "totals": [20, 30],
        "sums": [160, 240],
        "sim_count": 20,
        "a_priors_gamma": [0.5, 0.5],
        "b_priors_gamma": [0.5, 0.5],
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


@pytest.mark.parametrize("inp", DIRICHLET_POSTERIORS_INPUTS)
def test_dirichlet_posteriors(inp):
    all_pos = dirichlet_posteriors(
        inp["concentration"],
        inp["prior"],
        inp["sim_count"],
    )
    assert all_pos.shape == (inp["sim_count"], len(inp["concentration"]))


@pytest.mark.parametrize("inp", GAMMA_POSTERIORS_ALL_INPUTS)
def test_gamma_posteriors_all(inp):
    all_pos = gamma_posteriors_all(
        inp["totals"],
        inp["sums"],
        inp["sim_count"],
        inp["a_priors_gamma"],
        inp["b_priors_gamma"],
    )
    all_pos_shape = np.array(all_pos).shape
    assert all_pos_shape == (len(inp["totals"]), inp["sim_count"])
