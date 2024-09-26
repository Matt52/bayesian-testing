import pytest

from bayesian_testing.experiments import PoissonDataTest


@pytest.fixture
def poisson_test():
    pois = PoissonDataTest()
    pois.add_variant_data("A", [5, 5, 7, 1, 3, 3, 1, 1, 2, 0, 1, 3, 4, 2, 5])
    pois.add_variant_data("B", [2, 4, 3, 4, 6, 1, 3, 6, 4, 0, 3, 1, 2, 1])
    pois.add_variant_data_agg("C", 15, 49, a_prior=1, b_prior=2)
    pois.add_variant_data_agg("D", 10, 10)
    pois.add_variant_data_agg("D", 20, 20, replace=False)
    pois.add_variant_data_agg("D", 20, 20, replace=True)
    pois.delete_variant("D")
    return pois


def test_variants(poisson_test):
    assert poisson_test.variant_names == ["A", "B", "C"]


def test_totals(poisson_test):
    assert poisson_test.totals == [15, 14, 15]


def test_positives(poisson_test):
    assert poisson_test.sum_values == [43, 40, 49]


def test_a_priors(poisson_test):
    assert poisson_test.a_priors == [0.1, 0.1, 1]


def test_b_priors(poisson_test):
    assert poisson_test.b_priors == [0.1, 0.1, 2]


def test_probabs_of_being_best(poisson_test):
    pbbs = poisson_test.probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.30945, "B": 0.29665, "C": 0.3939}


def test_expected_loss(poisson_test):
    loss = poisson_test.expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 0.3936672, "B": 0.4144949, "C": 0.3109256}


def test_credible_intervals_95(poisson_test):
    ci = poisson_test.credible_intervals(sim_count=20000, seed=52)
    assert ci == {
        "A": [2.0742056, 3.7731115],
        "B": [2.0264899, 3.7822918],
        "C": [2.1895805, 3.8084984],
    }


def test_credible_intervals_99(poisson_test):
    ci = poisson_test.credible_intervals(sim_count=20000, seed=52, interval_alpha=0.99)
    assert ci == {
        "A": [1.8569798, 4.0897961],
        "B": [1.8082962, 4.1242607],
        "C": [1.9771075, 4.1434489],
    }


def test_evaluate(poisson_test):
    eval_report = poisson_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 15,
            "sum_values": 43,
            "observed_average": 2.86667,
            "posterior_mean": 2.8543,
            "credible_interval": [2.0742056, 3.7731115],
            "prob_being_best": 0.30945,
            "expected_loss": 0.3936672,
        },
        {
            "variant": "B",
            "totals": 14,
            "sum_values": 40,
            "observed_average": 2.85714,
            "posterior_mean": 2.84397,
            "credible_interval": [2.0264899, 3.7822918],
            "prob_being_best": 0.29665,
            "expected_loss": 0.4144949,
        },
        {
            "variant": "C",
            "totals": 15,
            "sum_values": 49,
            "observed_average": 3.26667,
            "posterior_mean": 2.94118,
            "credible_interval": [2.1895805, 3.8084984],
            "prob_being_best": 0.3939,
            "expected_loss": 0.3109256,
        },
    ]


def test_wrong_inputs():
    pois_test = PoissonDataTest()
    with pytest.raises(ValueError):
        pois_test.add_variant_data(10, [1, 2, 3])
    with pytest.raises(ValueError):
        pois_test.add_variant_data("A", [1, 2, 3], a_prior=-1)
    with pytest.raises(ValueError):
        pois_test.add_variant_data_agg("A", -1, 7)
    with pytest.raises(ValueError):
        pois_test.add_variant_data_agg("A", 1, -7)
    with pytest.raises(ValueError):
        pois_test.add_variant_data("A", [])
    with pytest.raises(ValueError):
        pois_test.add_variant_data("A", [1, 2, -3])
