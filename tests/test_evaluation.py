import numpy as np
import pytest

from bayesian_testing.metrics import (
    eval_bernoulli_agg,
    eval_normal_agg,
    eval_delta_lognormal_agg,
    eval_delta_normal_agg,
    eval_numerical_dirichlet_agg,
    eval_poisson_agg,
    eval_exponential_agg,
)

PBB_BERNOULLI_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([0.04185, 0.92235, 0.0358], [0.0030138, 6.06e-05, 0.0031649]),
    },
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
        },
        "expected_output": ([0.4594, 0.00925, 0.53135], [0.000781, 0.0037342, 0.0006299]),
    },
    {
        "input": {
            "totals": [100, 200],
            "successes": [80, 160],
            "sim_count": 10000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([0.4899, 0.5101], [0.0204051, 0.0182965]),
    },
    {
        "input": {
            "totals": [100, 100],
            "successes": [0, 0],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([0.5008, 0.4992], [0.0030829, 0.0031614]),
    },
    {
        "input": {
            "totals": [100],
            "successes": [77],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([], []),
    },
]

PBB_NORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [31000, 30000, 32000],
            "sums": [33669.629254438274, 32451.58924937506, 34745.69678322253],
            "sums_2": [659657.6891070933, 95284.82070196551, 260327.13931832163],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.43605, 0.19685, 0.3671], [0.0133512, 0.0179947, 0.0137618]),
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.345516947431, 10708.892428298526],
            "sums_2": [214614.35949718487, 31368.55305547222],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.94445, 0.05555], [0.0011338, 0.0753121]),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "sums": [0, 0, 0, 0],
            "sums_2": [0, 0, 0, 0],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": (
            [0.40785, 0.25105, 0.1928, 0.1483],
            [0.0058965, 0.0065083, 0.0066249, 0.0067183],
        ),
    },
    {
        "input": {
            "totals": [100],
            "sums": [0],
            "sums_2": [0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.35, 11446.35],
            "sums_2": [214614.36, 214614.36],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.5024, 0.4976], [0.0250157, 0.0256253]),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sums_2": [],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_DELTA_LOGNORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sum_logs": [3831.806394737816, 4211.72986767986, 4055.965234848171],
            "sum_logs_2": [11029.923165846496, 12259.51868396913, 12357.911862914],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.00015, 0.03345, 0.9664], [0.2209593, 0.1205541, 0.0008458]),
    },
    {
        "input": {
            "totals": [31000, 31000],
            "successes": [1550, 1550],
            "sum_logs": [4055.965234848171, 4055.965234848171],
            "sum_logs_2": [12357.911862914, 12357.911862914],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([0.5013, 0.4987], [0.028189, 0.0287233]),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "successes": [0, 0, 0, 0],
            "sum_logs": [0, 0, 0, 0],
            "sum_logs_2": [0, 0, 0, 0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([0.25, 0.25, 0.25, 0.25], [np.nan, np.nan, np.nan, np.nan]),
    },
    {
        "input": {
            "totals": [100],
            "successes": [10],
            "sum_logs": [0],
            "sum_logs_2": [0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sum_logs": [],
            "sum_logs_2": [],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_NUMERICAL_DIRICHLET_AGG_INPUTS = [
    {
        "input": {
            "states": [1, 2, 3, 4, 5, 6],
            "concentrations": [
                [10, 10, 10, 10, 20, 10],
                [10, 10, 10, 10, 10, 20],
                [10, 10, 10, 20, 10, 10],
            ],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.28205, 0.62335, 0.0946], [0.1999528, 0.0698306, 0.334045]),
    },
    {
        "input": {
            "states": [1, 2, 3],
            "concentrations": [[100, 100, 100]],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "states": [],
            "concentrations": [],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_POISSON_AGG_INPUTS = [
    {
        "input": {
            "totals": [3150, 3200, 3100],
            "sums": [10000, 10000, 10000],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([0.127, 0.00695, 0.86605], [0.0539495, 0.1042691, 0.0030418]),
    },
    {
        "input": {
            "totals": [3150, 3200, 3100],
            "sums": [10000, 10000, 10000],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
        },
        "expected_output": ([0.12775, 0.8656, 0.00665], [0.0532581, 0.0029385, 0.1041658]),
    },
    {
        "input": {
            "totals": [100],
            "sums": [77],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([], []),
    },
]

PBB_EXPONENTIAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [100, 90, 80],
            "sums": [1040.29884, 993.66883, 883.05801],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([0.1826, 0.4065, 0.4109], [1.5415303, 0.874454, 0.8968751]),
    },
    {
        "input": {
            "totals": [1000, 1000, 1000],
            "sums": [2288.69431, 2471.61961, 2745.7794],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
        },
        "expected_output": ([0.9594, 0.0406, 0.0], [0.0015906, 0.1860726, 0.4593495]),
    },
    {
        "input": {
            "totals": [100],
            "sums": [1007.25317],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
        },
        "expected_output": ([], []),
    },
]

PBB_DELTA_NORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000],
            "non_zeros": [10, 20],
            "sums": [102.02561, 273.02],
            "sums_2": [1700.8, 3567.5],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.0024, 0.9976], [4.4e-06, 0.0]),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "non_zeros": [0, 0, 0, 0],
            "sums": [0, 0, 0, 0],
            "sums_2": [0, 0, 0, 0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([0.25, 0.25, 0.25, 0.25], [np.nan, np.nan, np.nan, np.nan]),
    },
    {
        "input": {
            "totals": [100],
            "non_zeros": [10],
            "sums": [0],
            "sums_2": [0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [],
            "non_zeros": [],
            "sums": [],
            "sums_2": [],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]


@pytest.mark.parametrize("inp", PBB_BERNOULLI_AGG_INPUTS)
def test_eval_bernoulli_agg(inp):
    i = inp["input"]
    res = eval_bernoulli_agg(
        i["totals"],
        i["successes"],
        sim_count=i["sim_count"],
        seed=i["seed"],
        min_is_best=i["min_is_best"],
    )
    assert res == inp["expected_output"]


@pytest.mark.parametrize("inp", PBB_NORMAL_AGG_INPUTS)
def test_eval_normal_agg(inp):
    i = inp["input"]
    res = eval_normal_agg(
        i["totals"],
        i["sums"],
        i["sums_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res == inp["expected_output"]


def test_eval_normal_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_normal_agg([100, 100], [10, 10], [20, 20])
    run2 = eval_normal_agg([100, 100], [10, 10], [20, 20])
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_DELTA_LOGNORMAL_AGG_INPUTS)
def test_eval_delta_lognormal_agg(inp):
    i = inp["input"]
    res = eval_delta_lognormal_agg(
        i["totals"],
        i["successes"],
        i["sum_logs"],
        i["sum_logs_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res == inp["expected_output"]


def test_eval_delta_lognormal_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_delta_lognormal_agg([1000, 1000], [100, 100], [10, 10], [20, 20], sim_count=100000)
    run2 = eval_delta_lognormal_agg([1000, 1000], [100, 100], [10, 10], [20, 20], sim_count=100000)
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_NUMERICAL_DIRICHLET_AGG_INPUTS)
def test_eval_numerical_dirichlet_agg(inp):
    i = inp["input"]
    res = eval_numerical_dirichlet_agg(
        i["states"], i["concentrations"], sim_count=i["sim_count"], seed=i["seed"]
    )
    assert res == inp["expected_output"]


def test_eval_numerical_dirichlet_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_numerical_dirichlet_agg([1, 20], [[10, 10], [20, 20]])
    run2 = eval_numerical_dirichlet_agg([1, 20], [[10, 10], [20, 20]])
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_POISSON_AGG_INPUTS)
def test_eval_poisson_agg(inp):
    i = inp["input"]
    res = eval_poisson_agg(
        i["totals"],
        i["sums"],
        sim_count=i["sim_count"],
        seed=i["seed"],
        min_is_best=i["min_is_best"],
    )
    assert res == inp["expected_output"]


@pytest.mark.parametrize("inp", PBB_EXPONENTIAL_AGG_INPUTS)
def test_eval_exponential_agg(inp):
    i = inp["input"]
    res = eval_exponential_agg(
        i["totals"],
        i["sums"],
        sim_count=i["sim_count"],
        seed=i["seed"],
        min_is_best=i["min_is_best"],
    )
    assert res == inp["expected_output"]


@pytest.mark.parametrize("inp", PBB_DELTA_NORMAL_AGG_INPUTS)
def test_eval_delta_normal_agg(inp):
    i = inp["input"]
    res = eval_delta_normal_agg(
        i["totals"],
        i["non_zeros"],
        i["sums"],
        i["sums_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res == inp["expected_output"]
