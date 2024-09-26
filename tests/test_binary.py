import pytest

from bayesian_testing.experiments import BinaryDataTest


@pytest.fixture
def conv_test():
    cv = BinaryDataTest()
    cv.add_variant_data("A", [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    cv.add_variant_data("B", [0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    cv.add_variant_data_agg("C", 11, 2, a_prior=1, b_prior=2)
    cv.add_variant_data_agg("D", 10, 10)
    cv.add_variant_data_agg("D", 20, 20, replace=False)
    cv.add_variant_data_agg("D", 20, 20, replace=True)
    cv.delete_variant("D")
    return cv


def test_variants(conv_test):
    assert conv_test.variant_names == ["A", "B", "C"]


def test_totals(conv_test):
    assert conv_test.totals == [10, 10, 11]


def test_positives(conv_test):
    assert conv_test.positives == [3, 2, 2]


def test_a_priors(conv_test):
    assert conv_test.a_priors == [0.5, 0.5, 1]


def test_b_priors(conv_test):
    assert conv_test.b_priors == [0.5, 0.5, 2]


def test_probabs_of_being_best(conv_test):
    pbbs = conv_test.probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.57225, "B": 0.233, "C": 0.19475}


def test_expected_loss(conv_test):
    loss = conv_test.expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 0.0529281, "B": 0.1452113, "C": 0.1557502}


def test_credible_intervals_95(conv_test):
    ci = conv_test.credible_intervals(sim_count=20000, seed=52)
    assert ci == {
        "A": [0.0917579, 0.6028411],
        "B": [0.0442435, 0.5032699],
        "C": [0.0522996, 0.452392],
    }


def test_credible_intervals_99(conv_test):
    ci = conv_test.credible_intervals(sim_count=20000, seed=52, interval_alpha=0.99)
    assert ci == {
        "A": [0.0552614, 0.6892976],
        "B": [0.0214602, 0.6045644],
        "C": [0.0300364, 0.5320378],
    }


def test_evaluate(conv_test):
    eval_report = conv_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 10,
            "positives": 3,
            "positive_rate": 0.3,
            "posterior_mean": 0.31818,
            "credible_interval": [0.0917579, 0.6028411],
            "prob_being_best": 0.57225,
            "expected_loss": 0.0529281,
        },
        {
            "variant": "B",
            "totals": 10,
            "positives": 2,
            "positive_rate": 0.2,
            "posterior_mean": 0.22727,
            "credible_interval": [0.0442435, 0.5032699],
            "prob_being_best": 0.233,
            "expected_loss": 0.1452113,
        },
        {
            "variant": "C",
            "totals": 11,
            "positives": 2,
            "positive_rate": 0.18182,
            "posterior_mean": 0.21429,
            "credible_interval": [0.0522996, 0.452392],
            "prob_being_best": 0.19475,
            "expected_loss": 0.1557502,
        },
    ]


def test_wrong_inputs():
    cv = BinaryDataTest()
    with pytest.raises(ValueError):
        cv.add_variant_data(10, [1, 0, 1])
    with pytest.raises(ValueError):
        cv.add_variant_data("A", [1, 0, 1], a_prior=-1)
    with pytest.raises(ValueError):
        cv.add_variant_data_agg("A", -1, 7)
    with pytest.raises(ValueError):
        cv.add_variant_data_agg("A", 1, -7)
    with pytest.raises(ValueError):
        cv.add_variant_data("A", [])
    with pytest.raises(ValueError):
        cv.add_variant_data("A", [1, 2, 0])


def test_wrong_credible_interval_input(conv_test):
    with pytest.raises(ValueError):
        conv_test.evaluate(interval_alpha=2)
    with pytest.raises(ValueError):
        conv_test.evaluate(interval_alpha=-1)
