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
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.04185, 0.92235, 0.0358],
            [0.0030138, 6.06e-05, 0.0031649],
            [[0.0477826, 0.0526302], [0.0506933, 0.0555936], [0.0476604, 0.0524757]],
        ),
    },
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
            "interval_alpha": 0.99,
        },
        "expected_output": (
            [0.4594, 0.00925, 0.53135],
            [0.000781, 0.0037342, 0.0006299],
            [[0.0470873, 0.0534391], [0.0499116, 0.056421], [0.0469394, 0.0532695]],
        ),
    },
    {
        "input": {
            "totals": [100, 200],
            "successes": [80, 160],
            "sim_count": 10000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.5,
        },
        "expected_output": (
            [0.4899, 0.5101],
            [0.0204051, 0.0182965],
            [[0.7713375, 0.8248972], [0.7810789, 0.8179153]],
        ),
    },
    {
        "input": {
            "totals": [100, 100],
            "successes": [0, 0],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.5008, 0.4992],
            [0.0030829, 0.0031614],
            [[4.8e-06, 0.0252857], [4.8e-06, 0.0243717]],
        ),
    },
    {
        "input": {
            "totals": [100],
            "successes": [77],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.95,
        },
        "expected_output": ([1], [0], [[0.6810233, 0.8442006]]),
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.95,
        },
        "expected_output": ([], [], []),
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
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.43605, 0.19685, 0.3671],
            [0.0133512, 0.0179947, 0.0137618],
            [[1.0366696, 1.13634], [1.0652914, 1.0977888], [1.0574217, 1.1141581]],
        ),
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.345516947431, 10708.892428298526],
            "sums_2": [214614.35949718487, 31368.55305547222],
            "sim_count": 20000,
            "seed": 52,
            "interval_alpha": 0.99,
        },
        "expected_output": (
            [0.94445, 0.05555],
            [0.0011338, 0.0753121],
            [[1.0278553, 1.2601174], [1.0337017, 1.1071861]],
        ),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "sums": [0, 0, 0, 0],
            "sums_2": [0, 0, 0, 0],
            "sim_count": 20000,
            "seed": 52,
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.40785, 0.25105, 0.1928, 0.1483],
            [0.0058965, 0.0065083, 0.0066249, 0.0067183],
            [
                [-0.021071, 0.0232855],
                [-0.0101753, 0.0108701],
                [-0.0064358, 0.0070877],
                [-0.004795, 0.0052896],
            ],
        ),
    },
    {
        "input": {
            "totals": [100],
            "sums": [0],
            "sums_2": [0],
            "sim_count": 10000,
            "seed": 52,
            "interval_alpha": 0.95,
        },
        "expected_output": ([1], [0], [[-0.0019355, 0.0020896]]),
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.35, 11446.35],
            "sums_2": [214614.36, 214614.36],
            "sim_count": 20000,
            "seed": 52,
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.5024, 0.4976],
            [0.0250157, 0.0256253],
            [[1.0577297, 1.2331092], [1.0545188, 1.2327107]],
        ),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sums_2": [],
            "sim_count": 10000,
            "seed": 52,
            "interval_alpha": 0.95,
        },
        "expected_output": ([], [], []),
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
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.00015, 0.03345, 0.9664],
            [0.2209593, 0.1205541, 0.0008458],
            [[0.9065769, 1.0655643], [1.0046391, 1.1707248], [1.1085257, 1.3061752]],
        ),
    },
    {
        "input": {
            "totals": [31000, 31000],
            "successes": [1550, 1550],
            "sum_logs": [4055.965234848171, 4055.965234848171],
            "sum_logs_2": [12357.911862914, 12357.911862914],
            "sim_count": 10000,
            "seed": 52,
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.5013, 0.4987],
            [0.028189, 0.0287233],
            [[1.1227657, 1.2882371], [1.1210866, 1.2895949]],
        ),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "successes": [0, 0, 0, 0],
            "sum_logs": [0, 0, 0, 0],
            "sum_logs_2": [0, 0, 0, 0],
            "sim_count": 10000,
            "seed": 52,
            "interval_alpha": 0.5,
        },
        "expected_output": (
            [0.25, 0.25, 0.25, 0.25],
            [np.nan, np.nan, np.nan, np.nan],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ),
    },
    {
        "input": {
            "totals": [100],
            "successes": [10],
            "sum_logs": [0],
            "sum_logs_2": [0],
            "sim_count": 10000,
            "seed": 52,
            "interval_alpha": 0.95,
        },
        "expected_output": ([1], [0], [[0.051825, 0.1697968]]),
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sum_logs": [],
            "sum_logs_2": [],
            "sim_count": 10000,
            "seed": 52,
            "interval_alpha": 0.95,
        },
        "expected_output": ([], [], []),
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
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.28205, 0.62335, 0.0946],
            [0.1999528, 0.0698306, 0.334045],
            [[3.3214796, 4.0718396], [3.4218451, 4.2243033], [3.1984494, 3.9184425]],
        ),
    },
    {
        "input": {
            "states": [1, 2, 3],
            "concentrations": [[100, 100, 100]],
            "sim_count": 20000,
            "seed": 52,
            "interval_alpha": 0.9,
        },
        "expected_output": ([1], [0], [[1.9077157, 2.0908699]]),
    },
    {
        "input": {
            "states": [],
            "concentrations": [],
            "sim_count": 20000,
            "seed": 52,
            "interval_alpha": 0.9,
        },
        "expected_output": ([], [], []),
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
            "interval_alpha": 0.95,
        },
        "expected_output": (
            [0.127, 0.00695, 0.86605],
            [0.0539495, 0.1042691, 0.0030418],
            [[3.1132541, 3.2375641], [3.0635577, 3.1863114], [3.1634511, 3.2890376]],
        ),
    },
    {
        "input": {
            "totals": [3150, 3200, 3100],
            "sums": [10000, 10000, 10000],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.12775, 0.8656, 0.00665],
            [0.0532581, 0.0029385, 0.1041658],
            [[3.123135, 3.2276693], [3.0732817, 3.1764313], [3.1729959, 3.2788603]],
        ),
    },
    {
        "input": {
            "totals": [100],
            "sums": [77],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.75,
        },
        "expected_output": ([1], [0], [[0.6723231, 0.8727923]]),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.9,
        },
        "expected_output": ([], [], []),
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
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.1826, 0.4065, 0.4109],
            [1.5195025, 0.8380173, 0.8431285],
            [[8.8658129, 12.3263561], [9.3561749, 13.2588682], [9.2650625, 13.3809534]],
        ),
    },
    {
        "input": {
            "totals": [1000, 1000, 1000],
            "sums": [2288.69431, 2471.61961, 2745.7794],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.9594, 0.0406, 0.0],
            [0.0017238, 0.1865276, 0.4598496],
            [[2.1727503, 2.4111014], [2.3482046, 2.6066663], [2.6087576, 2.8941021]],
        ),
    },
    {
        "input": {
            "totals": [100],
            "sums": [1007.25317],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": True,
            "interval_alpha": 0.912,
        },
        "expected_output": ([1], [0], [[8.5325723, 11.9986705]]),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.9,
        },
        "expected_output": ([], [], []),
    },
]

PBB_DELTA_NORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [10000, 1000],
            "non_zeros": [1009, 111],
            "sums": [7026.30599, 801.53947],
            "sums_2": [49993.4988, 5891.6073],
            "sim_count": 20000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.08285, 0.91715],
            [0.1045921, 0.0026141],
            [[0.6683901, 0.7384471], [0.6897179, 0.9275315]],
        ),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "non_zeros": [0, 0, 0, 0],
            "sums": [0, 0, 0, 0],
            "sums_2": [0, 0, 0, 0],
            "sim_count": 10000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.9,
        },
        "expected_output": (
            [0.25, 0.25, 0.25, 0.25],
            [np.nan, np.nan, np.nan, np.nan],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        ),
    },
    {
        "input": {
            "totals": [100],
            "non_zeros": [10],
            "sums": [0],
            "sums_2": [0],
            "sim_count": 10000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.9,
        },
        "expected_output": ([1], [0], [[-0.0017847, 0.0020072]]),
    },
    {
        "input": {
            "totals": [],
            "non_zeros": [],
            "sums": [],
            "sums_2": [],
            "sim_count": 10000,
            "seed": 52,
            "min_is_best": False,
            "interval_alpha": 0.9,
        },
        "expected_output": ([], [], []),
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
        interval_alpha=i["interval_alpha"],
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
        interval_alpha=i["interval_alpha"],
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
        interval_alpha=i["interval_alpha"],
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
        interval_alpha=i["interval_alpha"],
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
        interval_alpha=i["interval_alpha"],
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
        min_is_best=i["min_is_best"],
        interval_alpha=i["interval_alpha"],
    )
    assert res == inp["expected_output"]
