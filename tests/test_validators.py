import pytest

from bayesian_testing.metrics.pbbs import validate_bernoulli_input
from bayesian_testing.utilities.common import check_list_lengths


def test_validate_bernoulli_input():
    validate_bernoulli_input([1, 2, 3], [1, 1, 1])
    validate_bernoulli_input([1, 2], [1, 1])
    validate_bernoulli_input([1], [1])


def test_validate_bernoulli_input_error():
    with pytest.raises(ValueError):
        validate_bernoulli_input([1, 2], [1])


def test_check_list_lengths():
    check_list_lengths([[1, 2, 3], [1, 1, 1], [2, 2, 2], [7, 7, 7]])
    check_list_lengths([[], [], []])


def test_check_list_lengths_error():
    with pytest.raises(ValueError):
        check_list_lengths([[1, 2, 3], [1, 1, 1], [2, 2, 2], [7, 7]])
