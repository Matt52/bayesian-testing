import pytest
from bayesian_testing.experiments import DeltaNormalDataTest


@pytest.fixture
def rev_test():
    rev = DeltaNormalDataTest()
    rev.add_variant_data_agg(
        name="A",
        totals=31500,
        non_zeros=10,
        sum_values=102.02561,
        sum_values_2=1700.8,
        a_prior_beta=1
    )
    rev.add_variant_data_agg(
        name="B",
        totals=32000,
        non_zeros=40,
        sum_values=273.02,
        sum_values_2=3567.5,
        a_prior_beta=0.02,
        m_prior=2,
        w_prior=0.02
    )

    rev.add_variant_data("C", [0, 10.7, -1, 8, 0, -3, 0, -10, 0, 11.22])
    rev.add_variant_data("C", [0, 10.7, -1, 8, 0, -3, 0, -10, 0, 11.22], replace=False)
    rev.add_variant_data("C", [0, 10.7, -1, 8, 0, -3, 0, -10, 0, 11.22], replace=True)
    rev.delete_variant("C")
    return rev


def test_variants(rev_test):
    assert rev_test.variant_names == ["A", "B"]


def test_totals(rev_test):
    assert rev_test.totals == [31500, 32000]


def test_non_zeros(rev_test):
    assert rev_test.non_zeros == [10, 40]


def test_sum_values(rev_test):
    assert rev_test.sum_values == [102.02561, 273.02]


def test_sum_values_2(rev_test):
    assert rev_test.sum_values_2 == [1700.8, 3567.5]


def test_a_priors_beta(rev_test):
    assert rev_test.a_priors_beta == [1, 0.02]


def test_b_priors_beta(rev_test):
    assert rev_test.b_priors_beta == [0.5, 0.5]


def test_m_priors(rev_test):
    assert rev_test.m_priors == [1, 2]


def test_a_priors_ig(rev_test):
    assert rev_test.a_priors_ig == [0, 0]


def test_b_priors_ig(rev_test):
    assert rev_test.b_priors_ig == [0, 0]


def test_w_priors(rev_test):
    assert rev_test.w_priors == [0.01, 0.02]


def test_probabs_of_being_best(rev_test):
    pbbs = rev_test.probabs_of_being_best(sim_count=20000, seed=152)
    assert pbbs == {'A': 0.0002, 'B': 0.9998}


def test_expected_loss(rev_test):
    loss = rev_test.expected_loss(sim_count=20000, seed=152)
    assert loss == {'A': 9.6e-06, 'B': 0.0}


def test_evaluate(rev_test):
    eval_report = rev_test.evaluate(sim_count=20000, seed=152)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 31500,
            "non_zeros": 10,
            "sum_values": 102.02561,
            "avg_values": 0.00324,
            "avg_non_zero_values": 10.20256,
            "prob_being_best": 0.0002,
            "expected_loss": 9.6e-06,
        },
        {
            "variant": "B",
            "totals": 32000,
            "non_zeros": 40,
            "sum_values": 273.02,
            "avg_values": 0.00853,
            "avg_non_zero_values": 6.8255,
            "prob_being_best": 0.9998,
            "expected_loss": 0.0,
        }
    ]
