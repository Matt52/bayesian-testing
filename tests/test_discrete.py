import pytest

from bayesian_testing.experiments import DiscreteDataTest


@pytest.fixture
def discrete_test():
    disc = DiscreteDataTest(categories=[1, 2, 3, 4, 5, 6])
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


def test_categories(discrete_test):
    assert discrete_test.categories == [1, 2, 3, 4, 5, 6]


def test_concentrations(discrete_test):
    assert discrete_test.concentrations == [
        [1, 5, 2, 5, 3, 4],
        [2, 0, 1, 3, 0, 4],
        [10, 10, 10, 10, 10, 10],
    ]


def test_probabs_of_being_best(discrete_test):
    pbbs = discrete_test.probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.35595, "B": 0.59325, "C": 0.0508}


def test_evaluate(discrete_test):
    eval_report = discrete_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "concentration": {1: 1.0, 2: 5.0, 3: 2.0, 4: 5.0, 5: 3.0, 6: 4.0},
            "average_value": 3.8,
            "prob_being_best": 0.35595,
        },
        {
            "variant": "B",
            "concentration": {1: 2.0, 2: 0.0, 3: 1.0, 4: 3.0, 5: 0.0, 6: 4.0},
            "average_value": 4.1,
            "prob_being_best": 0.59325,
        },
        {
            "variant": "C",
            "concentration": {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10},
            "average_value": 3.5,
            "prob_being_best": 0.0508,
        },
    ]


def test_non_numerical_categories_error():
    with pytest.raises(ValueError):
        DiscreteDataTest(categories=[1, 2.0, "3"])


def test_non_string_variant_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data_agg(1, [1, 2, 3, 8, 10, 7])


def test_length_mismatch_input_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data_agg("D", [1, 2, 3, 8, 10])


def test_empty_data_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data("D", [])


def test_non_existing_category_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data("D", [1, 2, 3, 5, 21])
