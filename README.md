[![Tests](https://github.com/Matt52/bayesian-testing/workflows/Tests/badge.svg)](https://github.com/Matt52/bayesian-testing/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/Matt52/bayesian-testing/branch/main/graph/badge.svg)](https://codecov.io/gh/Matt52/bayesian-testing)
[![PyPI](https://img.shields.io/pypi/v/bayesian-testing.svg)](https://pypi.org/project/bayesian-testing/)
# Bayesian A/B testing
`bayesian_testing` is a small package for a quick evaluation of A/B (or A/B/C/...) tests using Bayesian approach.

**Implemented tests:**
- [BinaryDataTest](bayesian_testing/experiments/binary.py)
  - **_input data_** - binary (`[0, 1, 0, ...]`)
  - convenient for conversion-like A/B testing
- [NormalDataTest](bayesian_testing/experiments/normal.py)
  - **_input data_** - normal data with unknown variance
  - convenient for normal data A/B testing
- [DeltaLognormalDataTest](bayesian_testing/experiments/delta_lognormal.py)
  - **_input data_** - lognormal data with zeros
  - convenient for revenue-like A/B testing
- [DiscreteDataTest](bayesian_testing/experiments/discrete.py)
  - **_input data_** - categorical data with numerical categories
  - convenient for discrete data A/B testing (e.g. dice rolls, star ratings, 1-10 ratings)

**Implemented evaluation metrics:**
- `Probability of Being Best`
  - probability of "being larger" from a data point of view
- `Expected Loss`
  - "risk" of choosing particular variant over other variants in the test
  - measured in the same units as a tested measure (e.g. positive rate or average value)

Evaluation metrics are calculated using simulations from posterior distributions (considering given data).


## Installation
`bayesian_testing` can be installed using pip:
```console
pip install bayesian_testing
```
Alternatively, you can clone the repository and use `poetry` manually:
```console
cd bayesian_testing
pip install poetry
poetry install
poetry shell
```

## Basic Usage
The primary features are classes:
- `BinaryDataTest`
- `NormalDataTest`
- `DeltaLognormalDataTest`
- `DiscreteDataTest`

In all cases, there are two methods to insert data:
- `add_variant_data` - adding raw data for a variant as a list of observations (or numpy 1-D array)
- `add_variant_data_agg` - adding aggregated variant data (this can be practical for a large data, as the
aggregation can be done on a database level)

Both methods for adding data are allowing specification of prior distributions
(see details in respective docstrings). Default prior setup should be sufficient for most of the cases
(e.g. cases with unknown priors or large amounts of data).

To get the results of the test, simply call method `evaluate`.

Probabilities of being best and expected loss are approximated using simulations, hence `evaluate` can return
slightly different values for different runs. To stabilize it, you can set `sim_count` parameter of `evaluate`
to higher value (default value is 20K), or even use `seed` parameter to fix it completely.


### BinaryDataTest
Class for Bayesian A/B test for binary-like data (e.g. conversions, successes, etc.).

**Example:**
```python
import numpy as np
from bayesian_testing.experiments import BinaryDataTest

# generating some random data
rng = np.random.default_rng(52)
# random 1x1500 array of 0/1 data with 5.2% probability for 1:
data_a = rng.binomial(n=1, p=0.052, size=1500)
# random 1x1200 array of 0/1 data with 6.7% probability for 1:
data_b = rng.binomial(n=1, p=0.067, size=1200)

# initialize a test:
test = BinaryDataTest()

# add variant using raw data (arrays of zeros and ones):
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
# priors can be specified like this (default for this test is a=b=1/2):
# test.add_variant_data("B", data_b, a_prior=1, b_prior=20)

# add variant using aggregated data (same as raw data with 950 zeros and 50 ones):
test.add_variant_data_agg("C", totals=1000, positives=50)

# evaluate test:
test.evaluate()
```

    [{'variant': 'A',
      'totals': 1500,
      'positives': 80,
      'positive_rate': 0.05333,
      'prob_being_best': 0.0669,
      'expected_loss': 0.0138774},
     {'variant': 'B',
      'totals': 1200,
      'positives': 80,
      'positive_rate': 0.06667,
      'prob_being_best': 0.8926,
      'expected_loss': 0.0004599},
     {'variant': 'C',
      'totals': 1000,
      'positives': 50,
      'positive_rate': 0.05,
      'prob_being_best': 0.0405,
      'expected_loss': 0.0170356}]

### NormalDataTest
Class for Bayesian A/B test for normal data.

**Example:**
```python
import numpy as np
from bayesian_testing.experiments import NormalDataTest

# generating some random data
rng = np.random.default_rng(21)
data_a = rng.normal(7.2, 2, 1000)
data_b = rng.normal(7.1, 2, 800)
data_c = rng.normal(7.0, 4, 500)

# initialize a test:
test = NormalDataTest()

# add variant using raw data:
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
# test.add_variant_data("C", data_c)

# add variant using aggregated data:
test.add_variant_data_agg("C", len(data_c), sum(data_c), sum(np.square(data_c)))

# evaluate test:
test.evaluate(sim_count=20000, seed=52)
```

    [{'variant': 'A',
      'totals': 1000,
      'sum_values': 7294.67901,
      'avg_values': 7.29468,
      'prob_being_best': 0.1707,
      'expected_loss': 0.1968735},
     {'variant': 'B',
      'totals': 800,
      'sum_values': 5685.86168,
      'avg_values': 7.10733,
      'prob_being_best': 0.00125,
      'expected_loss': 0.385112},
     {'variant': 'C',
      'totals': 500,
      'sum_values': 3736.91581,
      'avg_values': 7.47383,
      'prob_being_best': 0.82805,
      'expected_loss': 0.0169998}]

### DeltaLognormalDataTest
Class for Bayesian A/B test for delta-lognormal data (log-normal with zeros).
Delta-lognormal data is typical case of revenue per session data where many sessions have 0 revenue
but non-zero values are positive numbers with possible log-normal distribution.
To handle this data, the calculation is combining binary Bayes model for zero vs non-zero
"conversions" and log-normal model for non-zero values.

**Example:**
```python
import numpy as np
from bayesian_testing.experiments import DeltaLognormalDataTest

test = DeltaLognormalDataTest()

data_a = [7.1, 0.3, 5.9, 0, 1.3, 0.3, 0, 1.2, 0, 3.6, 0, 1.5, 2.2, 0, 4.9, 0, 0, 1.1, 0, 0, 7.1, 0, 6.9, 0]
data_b = [4.0, 0, 3.3, 19.3, 18.5, 0, 0, 0, 12.9, 0, 0, 0, 10.2, 0, 0, 23.1, 0, 3.7, 0, 0, 11.3, 10.0, 0, 18.3, 12.1]

# adding variant using raw data:
test.add_variant_data("A", data_a)
# test.add_variant_data("B", data_b)

# alternatively, variant can be also added using aggregated data:
# (looks more complicated but for large data it can be quite handy to move around only these sums)
test.add_variant_data_agg(
    name="B",
    totals=len(data_b),
    positives=sum(x > 0 for x in data_b),
    sum_values=sum(data_b),
    sum_logs=sum([np.log(x) for x in data_b if x > 0]),
    sum_logs_2=sum([np.square(np.log(x)) for x in data_b if x > 0])
)

# evaluate test:
test.evaluate(seed=21)
```

    [{'variant': 'A',
      'totals': 24,
      'positives': 13,
      'sum_values': 43.4,
      'avg_values': 1.80833,
      'avg_positive_values': 3.33846,
      'prob_being_best': 0.04815,
      'expected_loss': 4.0941101},
     {'variant': 'B',
      'totals': 25,
      'positives': 12,
      'sum_values': 146.7,
      'avg_values': 5.868,
      'avg_positive_values': 12.225,
      'prob_being_best': 0.95185,
      'expected_loss': 0.1588627}]

### DiscreteDataTest
Class for Bayesian A/B test for discrete data with finite number of numerical categories (states),
representing some value.
This test can be used for instance for dice rolls data (when looking for the "best" of multiple dice) or rating data
(e.g. 1-5 stars or 1-10 scale).

**Example:**
```python
import numpy as np
from bayesian_testing.experiments import DiscreteDataTest

# dice rolls data for 3 dice - A, B, C
data_a = [2, 5, 1, 4, 6, 2, 2, 6, 3, 2, 6, 3, 4, 6, 3, 1, 6, 3, 5, 6]
data_b = [1, 2, 2, 2, 2, 3, 2, 3, 4, 2]
data_c = [1, 3, 6, 5, 4]

# initialize a test with all possible states (i.e. numerical categories):
test = DiscreteDataTest(states=[1, 2, 3, 4, 5, 6])

# add variant using raw data:
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
test.add_variant_data("C", data_c)

# add variant using aggregated data:
# test.add_variant_data_agg("C", [1, 0, 1, 1, 1, 1]) # equivalent to rolls in data_c

# evaluate test:
test.evaluate(sim_count=20000, seed=52)
```

    [{'variant': 'A',
      'concentration': {1: 2.0, 2: 4.0, 3: 4.0, 4: 2.0, 5: 2.0, 6: 6.0},
      'average_value': 3.8,
      'prob_being_best': 0.54685,
      'expected_loss': 0.199953},
     {'variant': 'B',
      'concentration': {1: 1.0, 2: 6.0, 3: 2.0, 4: 1.0, 5: 0.0, 6: 0.0},
      'average_value': 2.3,
      'prob_being_best': 0.008,
      'expected_loss': 1.1826766},
     {'variant': 'C',
      'concentration': {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
      'average_value': 3.8,
      'prob_being_best': 0.44515,
      'expected_loss': 0.2870247}]

## Development
To set up a development environment, use [Poetry](https://python-poetry.org/) and [pre-commit](https://pre-commit.com):
```console
pip install poetry
poetry install
poetry run pre-commit install
```

## Roadmap

Test classes to be added:
- `PoissonDataTest`
- `ExponentialDataTest`

Metrics to be added:
- `Potential Value Remaining`

## References
- `bayesian_testing` package itself depends only on [numpy](https://numpy.org) package.
- Work on this package (including default priors selection) was inspired mainly by a Coursera
course [Bayesian Statistics: From Concept to Data Analysis](https://www.coursera.org/learn/bayesian-statistics).
