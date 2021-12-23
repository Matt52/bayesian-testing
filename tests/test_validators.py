import pytest
from bayes_ab_test.metrics.simulations import validate_bernoulli_input


def test_validate_bernoulli_input():
    validate_bernoulli_input([1, 2, 3], [1, 1, 1])
    validate_bernoulli_input([1, 2], [1, 1])
    validate_bernoulli_input([1], [1])


def test_validate_bernoulli_input_error():
    with pytest.raises(ValueError):
        validate_bernoulli_input([1, 2], [1])
