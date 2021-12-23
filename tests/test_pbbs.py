import numpy as np
import pytest

np.random.seed(52)

from bayes_ab_test.metrics import pbb_bernoulli_agg, pbb_lognormal_agg

PBB_BERNOULLI_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sim_count": 20000,
        },
        "expected_output": [0.0425, 0.9226, 0.0349],
    },
    {
        "input": {
            "totals": [100, 200],
            "successes": [80, 160],
            "sim_count": 10000,
        },
        "expected_output": [0.4977, 0.5023],
    },
]

PBB_LOGNORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sum_logs": [3831.806394737816, 4211.72986767986, 4055.965234848171],
            "sum_logs_2": [11029.923165846496, 12259.51868396913, 12357.911862914],
            "sim_count": 20000,
        },
        "expected_output": [0.0002, 0.03605, 0.96375],
    },
    {
        "input": {
            "totals": [31500, 31000],
            "successes": [1580, 1550],
            "sum_logs": [3831.806394737816, 4055.965234848171],
            "sum_logs_2": [11029.923165846496, 12357.911862914],
            "sim_count": 10000,
        },
        "expected_output": [0.0005, 0.9995],
    },
]


@pytest.mark.parametrize("inp", PBB_BERNOULLI_AGG_INPUTS)
def test_pbb_bernoulli_agg(inp):
    i = inp["input"]
    res = pbb_bernoulli_agg(i["totals"], i["successes"], sim_count=i["sim_count"])
    assert res == inp["expected_output"]


@pytest.mark.parametrize("inp", PBB_LOGNORMAL_AGG_INPUTS)
def test_pbb_lognormal_agg(inp):
    i = inp["input"]
    res = pbb_lognormal_agg(
        i["totals"], i["successes"], i["sum_logs"], i["sum_logs_2"], sim_count=i["sim_count"]
    )
    assert res == inp["expected_output"]
