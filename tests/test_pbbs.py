import pytest

from bayes_ab_test.metrics import (
    pbb_bernoulli_agg,
    pbb_normal_agg,
    pbb_delta_lognormal_agg,
)

PBB_BERNOULLI_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [0.04185, 0.92235, 0.0358],
    },
    {
        "input": {
            "totals": [100, 200],
            "successes": [80, 160],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": [0.4899, 0.5101],
    },
    {
        "input": {
            "totals": [100, 100],
            "successes": [0, 0],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [0.5008, 0.4992],
    },
    {
        "input": {
            "totals": [100],
            "successes": [77],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [1],
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [],
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
        "expected_output": [0.43605, 0.19685, 0.3671],
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.345516947431, 10708.892428298526],
            "sums_2": [214614.35949718487, 31368.55305547222],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [0.94445, 0.05555],
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "sums": [0, 0, 0, 0],
            "sums_2": [0, 0, 0, 0],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [0.40785, 0.25105, 0.1928, 0.1483],
    },
    {
        "input": {
            "totals": [100],
            "sums": [0],
            "sums_2": [0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": [1],
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.35, 11446.35],
            "sums_2": [214614.36, 214614.36],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": [0.5024, 0.4976],
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sums_2": [],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": [],
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
        "expected_output": [0.00015, 0.03345, 0.9664],
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
        "expected_output": [0.5013, 0.4987],
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
        "expected_output": [0.25, 0.25, 0.25, 0.25],
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
        "expected_output": [1],
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
        "expected_output": [],
    },
]


@pytest.mark.parametrize("inp", PBB_BERNOULLI_AGG_INPUTS)
def test_pbb_bernoulli_agg(inp):
    i = inp["input"]
    res = pbb_bernoulli_agg(
        i["totals"], i["successes"], sim_count=i["sim_count"], seed=i["seed"]
    )
    assert res == inp["expected_output"]


@pytest.mark.parametrize("inp", PBB_NORMAL_AGG_INPUTS)
def test_pbb_normal_agg(inp):
    i = inp["input"]
    res = pbb_normal_agg(
        i["totals"],
        i["sums"],
        i["sums_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res == inp["expected_output"]


def test_pbb_normal_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = pbb_normal_agg([100, 100], [10, 10], [20, 20])
    run2 = pbb_normal_agg([100, 100], [10, 10], [20, 20])
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_DELTA_LOGNORMAL_AGG_INPUTS)
def test_pbb_delta_lognormal_agg(inp):
    i = inp["input"]
    res = pbb_delta_lognormal_agg(
        i["totals"],
        i["successes"],
        i["sum_logs"],
        i["sum_logs_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res == inp["expected_output"]


def test_pbb_delta_lognormal_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = pbb_delta_lognormal_agg([1000, 1000], [100, 100], [10, 10], [20, 20])
    run2 = pbb_delta_lognormal_agg([1000, 1000], [100, 100], [10, 10], [20, 20])
    assert run1 != run2
