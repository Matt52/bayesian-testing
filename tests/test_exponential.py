import pytest

from bayesian_testing.experiments import ExponentialDataTest


@pytest.fixture
def exponential_test():
    expo = ExponentialDataTest()
    expo.add_variant_data(
        "A",
        [
            3.27,
            5.62,
            0.31,
            3.9,
            2.4,
            10.49,
            0.63,
            2.71,
            1.64,
            0.43,
            0.22,
            0.3,
            1.99,
            0.69,
            5.15,
            1.31,
            1.01,
            1.26,
            0.2,
            1.6,
        ],
    )
    expo.add_variant_data(
        "B",
        [
            0.28,
            0.18,
            0.13,
            4.79,
            1.07,
            0.69,
            5.75,
            2.07,
            9.67,
            2.79,
            0.18,
            5.8,
            12.81,
            2.33,
            2.28,
            1.56,
            4.18,
            1.47,
            1.67,
            0.98,
        ],
    )
    expo.add_variant_data_agg("C", 20, 72.27, a_prior=1, b_prior=2)
    expo.add_variant_data_agg("D", 100, 200)
    expo.add_variant_data_agg("D", 100, 220, replace=False)
    expo.add_variant_data_agg("D", 10, 20, replace=True)
    expo.delete_variant("D")
    return expo


def test_variants(exponential_test):
    assert exponential_test.variant_names == ["A", "B", "C"]


def test_totals(exponential_test):
    assert exponential_test.totals == [20, 20, 20]


def test_positives(exponential_test):
    assert exponential_test.sum_values == [45.13, 60.68, 72.27]


def test_a_priors(exponential_test):
    assert exponential_test.a_priors == [0.1, 0.1, 1]


def test_b_priors(exponential_test):
    assert exponential_test.b_priors == [0.1, 0.1, 2]


def test_probabs_of_being_best(exponential_test):
    pbbs = exponential_test.probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.0414, "B": 0.29885, "C": 0.65975}


def test_expected_loss(exponential_test):
    loss = exponential_test.expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 1.574346, "B": 0.7704876, "C": 0.2741126}


def test_evaluate(exponential_test):
    eval_report = exponential_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 20,
            "sum_values": 45.13,
            "observed_average": 2.2565,
            "posterior_mean": 2.25025,
            "prob_being_best": 0.0414,
            "expected_loss": 1.574346,
        },
        {
            "variant": "B",
            "totals": 20,
            "sum_values": 60.68,
            "observed_average": 3.034,
            "posterior_mean": 3.02388,
            "prob_being_best": 0.29885,
            "expected_loss": 0.7704876,
        },
        {
            "variant": "C",
            "totals": 20,
            "sum_values": 72.27,
            "observed_average": 3.6135,
            "posterior_mean": 3.53667,
            "prob_being_best": 0.65975,
            "expected_loss": 0.2741126,
        },
    ]


def test_wrong_inputs():
    pois = ExponentialDataTest()
    with pytest.raises(ValueError):
        pois.add_variant_data(10, [1, 2, 3])
    with pytest.raises(ValueError):
        pois.add_variant_data("A", [1, 2, 3], a_prior=-1)
    with pytest.raises(ValueError):
        pois.add_variant_data_agg("A", -1, 7)
    with pytest.raises(ValueError):
        pois.add_variant_data_agg("A", 1, -7)
    with pytest.raises(ValueError):
        pois.add_variant_data("A", [])
    with pytest.raises(ValueError):
        pois.add_variant_data("A", [1, 2, -3])
