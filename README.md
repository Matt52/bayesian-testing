[![Tests](https://github.com/Matt52/bayesian-testing/workflows/Tests/badge.svg)](https://github.com/Matt52/bayesian-testing/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/Matt52/bayesian-testing/branch/main/graph/badge.svg)](https://codecov.io/gh/Matt52/bayesian-testing)
[![PyPI](https://img.shields.io/pypi/v/bayesian-testing.svg)](https://pypi.org/project/bayesian-testing/)
# Bayesian A/B testing
`bayesian_testing` is a small package for a quick evaluation of A/B (or A/B/C/...) tests using Bayesian approach.

**Implemented tests:**
- [BinaryDataTest](bayesian_testing/experiments/binary.py)
  - **_input data_** - binary data (`[0, 1, 0, ...]`)
  - designed for conversion-like A/B testing
- [NormalDataTest](bayesian_testing/experiments/normal.py)
  - **_input data_** - normal data with unknown variance
  - designed for normal data A/B testing
- [DeltaLognormalDataTest](bayesian_testing/experiments/delta_lognormal.py)
  - **_input data_** - lognormal data with zeros
  - designed for revenue-like A/B testing
- [DeltaNormalDataTest](bayesian_testing/experiments/delta_normal.py)
  - **_input data_** - normal data with zeros
  - designed for profit-like A/B testing
- [DiscreteDataTest](bayesian_testing/experiments/discrete.py)
  - **_input data_** - categorical data with numerical categories
  - designed for discrete data A/B testing (e.g. dice rolls, star ratings, 1-10 ratings)
- [PoissonDataTest](bayesian_testing/experiments/poisson.py)
  - **_input data_** - observations of non-negative integers (`[1, 0, 3, ...]`)
  - designed for poisson data A/B testing

**Implemented evaluation metrics:**
- `Probability of Being Best`
  - probability that a given variant is best among all variants
  - by default, `best` is equivalent to `greatest` (from a data/metric point of view),
however it is possible to change it using `min_is_best=True` in the evaluation method
(this can be useful if we try to find the variant while minimizing tested measure)
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
- `DeltaNormalDataTest`
- `DiscreteDataTest`
- `PoissonDataTest`

All test classes support two methods to insert the data:
- `add_variant_data` - adding raw data for a variant as a list of observations (or numpy 1-D array)
- `add_variant_data_agg` - adding aggregated variant data (this can be practical for a large data, as the
aggregation can be done already on a database level)

Both methods for adding data allow specification of prior distributions
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
results = test.evaluate()
results # print(pd.DataFrame(results).to_markdown(tablefmt="grid", index=False))
```

    +---------+--------+-----------+---------------+----------------+-----------------+---------------+
    | variant | totals | positives | positive_rate | posterior_mean | prob_being_best | expected_loss |
    +=========+========+===========+===============+================+=================+===============+
    | A       |   1500 |        80 |       0.05333 |        0.05363 |         0.067   |     0.0138102 |
    +---------+--------+-----------+---------------+----------------+-----------------+---------------+
    | B       |   1200 |        80 |       0.06667 |        0.06703 |         0.88975 |     0.0004622 |
    +---------+--------+-----------+---------------+----------------+-----------------+---------------+
    | C       |   1000 |        50 |       0.05    |        0.05045 |         0.04325 |     0.0169686 |
    +---------+--------+-----------+---------------+----------------+-----------------+---------------+

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
results = test.evaluate(sim_count=20000, seed=52, min_is_best=False)
results # print(pd.DataFrame(results).to_markdown(tablefmt="grid", index=False))
```

    +---------+--------+------------+------------+----------------+-----------------+---------------+
    | variant | totals | sum_values | avg_values | posterior_mean | prob_being_best | expected_loss |
    +=========+========+============+============+================+=================+===============+
    | A       |   1000 |    7294.68 |    7.29468 |        7.29462 |         0.1707  |     0.196874  |
    +---------+--------+------------+------------+----------------+-----------------+---------------+
    | B       |    800 |    5685.86 |    7.10733 |        7.10725 |         0.00125 |     0.385112  |
    +---------+--------+------------+------------+----------------+-----------------+---------------+
    | C       |    500 |    3736.92 |    7.47383 |        7.4737  |         0.82805 |     0.0169998 |
    +---------+--------+------------+------------+----------------+-----------------+---------------+

### DeltaLognormalDataTest
Class for Bayesian A/B test for delta-lognormal data (log-normal with zeros).
Delta-lognormal data is typical case of revenue per session data where many sessions have 0 revenue
but non-zero values are positive values with possible log-normal distribution.
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

# alternatively, variant can be also added using aggregated data
# (looks more complicated, but it can be quite handy for a large data):
test.add_variant_data_agg(
    name="B",
    totals=len(data_b),
    positives=sum(x > 0 for x in data_b),
    sum_values=sum(data_b),
    sum_logs=sum([np.log(x) for x in data_b if x > 0]),
    sum_logs_2=sum([np.square(np.log(x)) for x in data_b if x > 0])
)

# evaluate test:
results = test.evaluate(seed=21)
results # print(pd.DataFrame(results).to_markdown(tablefmt="grid", index=False))
```

    +---------+--------+-----------+------------+------------+---------------------+-----------------+---------------+
    | variant | totals | positives | sum_values | avg_values | avg_positive_values | prob_being_best | expected_loss |
    +=========+========+===========+============+============+=====================+=================+===============+
    | A       |     24 |        13 |       43.4 |    1.80833 |             3.33846 |         0.04815 |      4.09411  |
    +---------+--------+-----------+------------+------------+---------------------+-----------------+---------------+
    | B       |     25 |        12 |      146.7 |    5.868   |            12.225   |         0.95185 |      0.158863 |
    +---------+--------+-----------+------------+------------+---------------------+-----------------+---------------+

***Note**: Alternatively, `DeltaNormalDataTest` can be used for a case when
conversions are not necessarily positive values.*

### DiscreteDataTest
Class for Bayesian A/B test for discrete data with finite number of numerical categories (states),
representing some value.
This test can be used for instance for dice rolls data (when looking for the "best" of multiple dice) or rating data
(e.g. 1-5 stars or 1-10 scale).

**Example:**
```python
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
results = test.evaluate(sim_count=20000, seed=52, min_is_best=False)
results # print(pd.DataFrame(results).to_markdown(tablefmt="grid", index=False))
```

    +---------+--------------------------------------------------+---------------+-----------------+---------------+
    | variant | concentration                                    | average_value | prob_being_best | expected_loss |
    +=========+==================================================+===============+=================+===============+
    | A       | {1: 2.0, 2: 4.0, 3: 4.0, 4: 2.0, 5: 2.0, 6: 6.0} |           3.8 |         0.54685 |      0.199953 |
    +---------+--------------------------------------------------+---------------+-----------------+---------------+
    | B       | {1: 1.0, 2: 6.0, 3: 2.0, 4: 1.0, 5: 0.0, 6: 0.0} |           2.3 |         0.008   |      1.18268  |
    +---------+--------------------------------------------------+---------------+-----------------+---------------+
    | C       | {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0} |           3.8 |         0.44515 |      0.287025 |
    +---------+--------------------------------------------------+---------------+-----------------+---------------+

### PoissonDataTest
Class for Bayesian A/B test for poisson data.

**Example:**
```python
from bayesian_testing.experiments import PoissonDataTest

# goals received - so less is better (duh...)
psg_goals_against = [0, 2, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3, 1, 0]
city_goals_against = [0, 0, 3, 2, 0, 1, 0, 3, 0, 1, 1, 0, 1, 2]
bayern_goals_against = [1, 0, 0, 1, 1, 2, 1, 0, 2, 0, 0, 2, 2, 1, 0]

# initialize a test:
test = PoissonDataTest()

# add variant using raw data:
test.add_variant_data('psg', psg_goals_against)

# example with specific priors
# ("b_prior" as an effective sample size, and "a_prior/b_prior" as a prior mean):
test.add_variant_data('city', city_goals_against, a_prior=3, b_prior=1)
# test.add_variant_data('bayern', bayern_goals_against)

# add variant using aggregated data:
test.add_variant_data_agg("bayern", len(bayern_goals_against), sum(bayern_goals_against))

# evaluate test (since fewer goals is better, we explicitly set min_is_best to True)
results = test.evaluate(sim_count=20000, seed=52, min_is_best=True)
results # print(pd.DataFrame(results).to_markdown(tablefmt="grid", index=False))
```

    +---------+--------+------------+------------------+----------------+-----------------+---------------+
    | variant | totals | sum_values | observed_average | posterior_mean | prob_being_best | expected_loss |
    +=========+========+============+==================+================+=================+===============+
    | psg     |     15 |          9 |          0.6     |        0.60265 |         0.78175 |     0.0369998 |
    +---------+--------+------------+------------------+----------------+-----------------+---------------+
    | city    |     14 |         14 |          1       |        1.13333 |         0.0344  |     0.562055  |
    +---------+--------+------------+------------------+----------------+-----------------+---------------+
    | bayern  |     15 |         13 |          0.86667 |        0.86755 |         0.18385 |     0.300335  |
    +---------+--------+------------+------------------+----------------+-----------------+---------------+

_note: Since we set `min_is_best=True` (because received goals are "bad"), probability and loss are in a favor of variants with lower posterior means._

## Development
To set up a development environment, use [Poetry](https://python-poetry.org/) and [pre-commit](https://pre-commit.com):
```console
pip install poetry
poetry install
poetry run pre-commit install
```

## Roadmap

Test classes to be added:
- `ExponentialDataTest`

Metrics to be added:
- `Potential Value Remaining`

## References
- `bayesian_testing` package itself depends only on [numpy](https://numpy.org) package.
- Work on this package (including default priors selection) was inspired mainly by a Coursera
course [Bayesian Statistics: From Concept to Data Analysis](https://www.coursera.org/learn/bayesian-statistics).
