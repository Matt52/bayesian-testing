import pytest

from bayesian_testing.experiments import NormalDataTest


@pytest.fixture
def norm_test():
    norm = NormalDataTest()
    norm.add_variant_data(
        "A",
        [
            11.8,
            12.2,
            12.4,
            9.5,
            2.2,
            3.3,
            16.2,
            4.9,
            12.4,
            6.8,
            8.7,
            9.8,
            5.4,
            9.0,
            15.0,
            12.3,
            9.6,
            12.5,
            9.1,
            10.2,
        ],
        m_prior=9,
    )
    norm.add_variant_data(
        "B",
        [
            10.6,
            5.1,
            9.4,
            11.2,
            2.0,
            13.4,
            14.1,
            15.4,
            16.3,
            11.7,
            7.3,
            6.8,
            8.2,
            16.2,
            10.8,
            7.1,
            12.2,
            11.2,
        ],
        w_prior=0.03,
    )
    norm.add_variant_data(
        "C",
        [
            25.3,
            10.3,
            24.7,
            -8.1,
            8.4,
            10.3,
            14.8,
            13.4,
            11.5,
            -4.7,
            5.3,
            7.4,
            17.2,
            15.4,
            13.0,
            12.9,
            19.2,
            11.6,
            0.4,
            5.7,
            23.5,
            15.2,
        ],
        b_prior_ig=2,
    )
    norm.add_variant_data_agg("A", 20, 193.3, 2127.71, replace=False)
    norm.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22])
    norm.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=False)
    norm.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=True)
    norm.delete_variant("D")
    return norm


def test_variants(norm_test):
    assert norm_test.variant_names == ["A", "B", "C"]


def test_totals(norm_test):
    assert norm_test.totals == [40, 18, 22]


def test_sum_values(norm_test):
    assert norm_test.sum_values == [386.6, 188.99999999999997, 252.69999999999996]


def test_sum_values_2(norm_test):
    assert norm_test.sum_values_2 == [4255.42, 2244.8200000000006, 4421.87]


def test_m_priors(norm_test):
    assert norm_test.m_priors == [9, 1, 1]


def test_a_priors_ig(norm_test):
    assert norm_test.a_priors_ig == [0, 0, 0]


def test_b_priors_ig(norm_test):
    assert norm_test.b_priors_ig == [0, 0, 2]


def test_w_priors(norm_test):
    assert norm_test.w_priors == [0.01, 0.03, 0.01]


def test_probabs_of_being_best(norm_test):
    pbbs = norm_test.probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.05105, "B": 0.27935, "C": 0.6696}


def test_expected_loss(norm_test):
    loss = norm_test.expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 2.2696341, "B": 1.4580033, "C": 0.4464154}


def test_credible_intervals_95(norm_test):
    ci = norm_test.credible_intervals(sim_count=20000, seed=52)
    assert ci == {
        "A": [8.5300072, 10.8231841],
        "B": [8.5577171, 12.3448628],
        "C": [7.8915125, 15.1179586],
    }


def test_credible_intervals_99(norm_test):
    ci = norm_test.credible_intervals(sim_count=20000, seed=52, interval_alpha=0.99)
    assert ci == {
        "A": [8.1196181, 11.2023581],
        "B": [7.8792145, 13.0964176],
        "C": [6.5669908, 16.5226358],
    }


def test_evaluate(norm_test):
    eval_report = norm_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 40,
            "sum_values": 386.6,
            "avg_values": 9.665,
            "posterior_mean": 9.66483,
            "prob_being_best": 0.05105,
            "expected_loss": 2.2696341,
            "credible_interval": [8.5300072, 10.8231841],
        },
        {
            "variant": "B",
            "totals": 18,
            "sum_values": 189.0,
            "avg_values": 10.5,
            "posterior_mean": 10.48419,
            "prob_being_best": 0.27935,
            "expected_loss": 1.4580033,
            "credible_interval": [8.5577171, 12.3448628],
        },
        {
            "variant": "C",
            "totals": 22,
            "sum_values": 252.7,
            "avg_values": 11.48636,
            "posterior_mean": 11.4816,
            "prob_being_best": 0.6696,
            "expected_loss": 0.4464154,
            "credible_interval": [7.8915125, 15.1179586],
        },
    ]
