import pytest

from bayes_ab_test.experiments import DeltaLognormalDataTest


@pytest.fixture
def rev_test():
    rev = DeltaLognormalDataTest()
    rev.add_variant_data("A", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], a_prior_beta=1)
    rev.add_variant_data(
        "B", [0, 0, 0, 11.3, 0, 0, 0, 0, 0, 9.1], m_prior=2, w_prior=0.02
    )
    rev.add_variant_data_agg(
        "C",
        11,
        3,
        23.1,
        6.079523500198114,
        12.409452935840312,
        a_prior_ig=1,
        b_prior_ig=2,
    )
    rev.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22])
    rev.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=False)
    rev.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=True)
    rev.delete_variant("D")
    return rev


def test_variants(rev_test):
    assert rev_test.variant_names == ["A", "B", "C"]


def test_totals(rev_test):
    assert rev_test.totals == [10, 10, 11]


def test_positives(rev_test):
    assert rev_test.positives == [3, 2, 3]


def test_sum_values(rev_test):
    assert rev_test.sum_values == [29.92, 20.4, 23.1]


def test_sum_logs(rev_test):
    assert [round(i, 5) for i in rev_test.sum_logs] == [6.86738, 4.63308, 6.07952]


def test_sum_logs_2(rev_test):
    assert [round(i, 5) for i in rev_test.sum_logs_2] == [15.7874, 10.75614, 12.40945]


def test_a_priors_beta(rev_test):
    assert rev_test.a_priors_beta == [1, 0.5, 0.5]


def test_b_priors_beta(rev_test):
    assert rev_test.b_priors_beta == [0.5, 0.5, 0.5]


def test_m_priors(rev_test):
    assert rev_test.m_priors == [1, 2, 1]


def test_a_priors_ig(rev_test):
    assert rev_test.a_priors_ig == [0, 0, 1]


def test_b_priors_ig(rev_test):
    assert rev_test.b_priors_ig == [0, 0, 2]


def test_w_priors(rev_test):
    assert rev_test.w_priors == [0.01, 0.02, 0.01]


def test_probabs_of_being_best(rev_test):
    pbbs = rev_test.probabs_of_being_best(sim_count=20000, seed=152)
    assert pbbs == {"A": 0.3827, "B": 0.13765, "C": 0.47965}


def test_evaluate(rev_test):
    eval_report = rev_test.evaluate(sim_count=20000, seed=152)
    print(eval_report)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 10,
            "positives": 3,
            "sum_values": 29.92,
            "avg_values": 2.992,
            "avg_positive_values": 9.97333,
            "prob_being_best": 0.3827,
        },
        {
            "variant": "B",
            "totals": 10,
            "positives": 2,
            "sum_values": 20.4,
            "avg_values": 2.04,
            "avg_positive_values": 10.2,
            "prob_being_best": 0.13765,
        },
        {
            "variant": "C",
            "totals": 11,
            "positives": 3,
            "sum_values": 23.1,
            "avg_values": 2.1,
            "avg_positive_values": 7.7,
            "prob_being_best": 0.47965,
        },
    ]
