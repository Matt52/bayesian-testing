import pytest

from bayesian_testing.experiments import DiscreteDataTest


@pytest.fixture
def discrete_test():
    disc = DiscreteDataTest(states=[1, 2, 3, 4, 5, 6])
    disc.add_variant_data("A", [6, 5, 4, 4, 4, 2, 5, 4, 2, 1, 2, 5, 4, 6, 2, 3, 6, 2, 3, 6])
    disc.add_variant_data("B", [4, 6, 3, 6, 4, 6, 6, 1, 4, 1])
    disc.add_variant_data_agg("C", [10, 10, 10, 10, 10, 10], prior=[100, 100, 100, 100, 100, 100])
    disc.add_variant_data_agg("D", [1, 2, 3, 8, 10, 7])
    disc.add_variant_data_agg("D", [1, 2, 3, 8, 10, 6], replace=False)
    disc.add_variant_data_agg("D", [1, 2, 3, 8, 10, 6], replace=True)
    disc.delete_variant("D")
    return disc


def test_variants(discrete_test):
    assert discrete_test.variant_names == ["A", "B", "C"]


def test_states(discrete_test):
    assert discrete_test.states == [1, 2, 3, 4, 5, 6]


def test_concentrations(discrete_test):
    assert discrete_test.concentrations == [
        [1, 5, 2, 5, 3, 4],
        [2, 0, 1, 3, 0, 4],
        [10, 10, 10, 10, 10, 10],
    ]


def test_probabs_of_being_best(discrete_test):
    pbbs = discrete_test.probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.35595, "B": 0.59325, "C": 0.0508}


def test_expected_loss(discrete_test):
    loss = discrete_test.expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 0.3053921, "B": 0.1560257, "C": 0.5328904}


def test_credible_intervals_95(discrete_test):
    ci = discrete_test.credible_intervals(sim_count=20000, seed=52)
    assert ci == {
        "A": [3.122705, 4.3265574],
        "B": [2.9826238, 4.7094185],
        "C": [3.3681015, 3.6302274],
    }


def test_credible_intervals_99(discrete_test):
    ci = discrete_test.credible_intervals(sim_count=20000, seed=52, interval_alpha=0.99)
    assert ci == {
        "A": [2.9260719, 4.5245231],
        "B": [2.7013326, 4.9277036],
        "C": [3.3281699, 3.6751105],
    }


def test_evaluate(discrete_test):
    eval_report = discrete_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "concentration": {1: 1.0, 2: 5.0, 3: 2.0, 4: 5.0, 5: 3.0, 6: 4.0},
            "average_value": 3.8,
            "posterior_mean": 3.73077,
            "credible_interval": [3.122705, 4.3265574],
            "prob_being_best": 0.35595,
            "expected_loss": 0.3053921,
        },
        {
            "variant": "B",
            "concentration": {1: 2.0, 2: 0.0, 3: 1.0, 4: 3.0, 5: 0.0, 6: 4.0},
            "average_value": 4.1,
            "posterior_mean": 3.875,
            "credible_interval": [2.9826238, 4.7094185],
            "prob_being_best": 0.59325,
            "expected_loss": 0.1560257,
        },
        {
            "variant": "C",
            "concentration": {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10},
            "average_value": 3.5,
            "posterior_mean": 3.5,
            "credible_interval": [3.3681015, 3.6302274],
            "prob_being_best": 0.0508,
            "expected_loss": 0.5328904,
        },
    ]


def test_non_numerical_states_error():
    with pytest.raises(ValueError):
        DiscreteDataTest(states=[1, 2.0, "3"])


def test_non_string_variant_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data_agg(1, [1, 2, 3, 8, 10, 7])


def test_length_mismatch_input_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data_agg("D", [1, 2, 3, 8, 10])


def test_empty_data_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data("D", [])


def test_non_existing_state_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data("D", [1, 2, 3, 5, 21])
