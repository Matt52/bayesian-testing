import pytest
from bayesian_testing.experiments import DeltaNormalDataTest


@pytest.fixture
def delta_norm_test():
    delta_norm = DeltaNormalDataTest()
    delta_norm.add_variant_data_agg(
        name="A",
        totals=31500,
        non_zeros=10,
        sum_values=102.02561,
        sum_values_2=1700.8,
        a_prior_beta=1,
    )
    delta_norm.add_variant_data_agg(
        name="B",
        totals=32000,
        non_zeros=40,
        sum_values=273.02,
        sum_values_2=3567.5,
        a_prior_beta=0.02,
        m_prior=2,
        w_prior=0.02,
    )

    delta_norm.add_variant_data("C", [0, 10.7, -1, 8, 0, -3, 0, -10, 0, 11.22])
    delta_norm.add_variant_data("C", [0, 10.7, -1, 8, 0, -3, 0, -10, 0, 11.22], replace=False)
    delta_norm.add_variant_data("C", [0, 10.7, -1, 8, 0, -3, 0, -10, 0, 11.22], replace=True)
    delta_norm.delete_variant("C")
    return delta_norm


def test_variants(delta_norm_test):
    assert delta_norm_test.variant_names == ["A", "B"]


def test_totals(delta_norm_test):
    assert delta_norm_test.totals == [31500, 32000]


def test_non_zeros(delta_norm_test):
    assert delta_norm_test.non_zeros == [10, 40]


def test_sum_values(delta_norm_test):
    assert delta_norm_test.sum_values == [102.02561, 273.02]


def test_sum_values_2(delta_norm_test):
    assert delta_norm_test.sum_values_2 == [1700.8, 3567.5]


def test_a_priors_beta(delta_norm_test):
    assert delta_norm_test.a_priors_beta == [1, 0.02]


def test_b_priors_beta(delta_norm_test):
    assert delta_norm_test.b_priors_beta == [0.5, 0.5]


def test_m_priors(delta_norm_test):
    assert delta_norm_test.m_priors == [1, 2]


def test_a_priors_ig(delta_norm_test):
    assert delta_norm_test.a_priors_ig == [0, 0]


def test_b_priors_ig(delta_norm_test):
    assert delta_norm_test.b_priors_ig == [0, 0]


def test_w_priors(delta_norm_test):
    assert delta_norm_test.w_priors == [0.01, 0.02]


def test_probabs_of_being_best(delta_norm_test):
    pbbs = delta_norm_test.probabs_of_being_best(sim_count=20000, seed=152)
    assert pbbs == {"A": 0.02235, "B": 0.97765}


def test_expected_loss(delta_norm_test):
    loss = delta_norm_test.expected_loss(sim_count=20000, seed=152)
    assert loss == {"A": 0.005, "B": 2.46e-05}


def test_credible_intervals_95(delta_norm_test):
    ci = delta_norm_test.credible_intervals(sim_count=20000, seed=152)
    assert ci == {
        "A": [0.0011935, 0.0070944],
        "B": [0.0051651, 0.0125917],
    }


def test_credible_intervals_99(delta_norm_test):
    ci = delta_norm_test.credible_intervals(sim_count=20000, seed=152, interval_alpha=0.99)
    assert ci == {
        "A": [0.0006048, 0.0087352],
        "B": [0.0043509, 0.0142946],
    }


def test_evaluate(delta_norm_test):
    eval_report = delta_norm_test.evaluate(sim_count=20000, seed=152)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 31500,
            "non_zeros": 10,
            "sum_values": 102.02561,
            "avg_values": 0.00324,
            "avg_non_zero_values": 10.20256,
            "posterior_mean": 0.00356,
            "credible_interval": [0.0011935, 0.0070944],
            "prob_being_best": 0.02235,
            "expected_loss": 0.005,
        },
        {
            "variant": "B",
            "totals": 32000,
            "non_zeros": 40,
            "sum_values": 273.02,
            "avg_values": 0.00853,
            "avg_non_zero_values": 6.8255,
            "posterior_mean": 0.00853,
            "credible_interval": [0.0051651, 0.0125917],
            "prob_being_best": 0.97765,
            "expected_loss": 2.46e-05,
        },
    ]


def test_wrong_inputs():
    dn_test = DeltaNormalDataTest()
    with pytest.raises(ValueError):
        dn_test.add_variant_data(10, [1, 2, 3])
    with pytest.raises(ValueError):
        dn_test.add_variant_data("A", [1, 2, 3], a_prior_beta=-1)
    with pytest.raises(ValueError):
        dn_test.add_variant_data_agg("A", 2, 3, 6, 21)
    with pytest.raises(ValueError):
        dn_test.add_variant_data_agg("A", 1, -7, 6, 21)
    with pytest.raises(ValueError):
        dn_test.add_variant_data("A", [])
    with pytest.raises(ValueError):
        dn_test.add_variant_data("A", [0, 0, 0])
    with pytest.raises(ValueError):
        dn_test.add_variant_data("C", [0, 10.7, -1], a_prior_ig=-1)
