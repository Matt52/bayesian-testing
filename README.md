[![Tests](https://github.com/Matt52/bayesian-testing/workflows/Tests/badge.svg)](https://github.com/Matt52/bayesian-testing/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/Matt52/bayesian-testing/branch/main/graph/badge.svg)](https://codecov.io/gh/Matt52/bayesian-testing)
[![PyPI](https://img.shields.io/pypi/v/bayesian-testing.svg)](https://pypi.org/project/bayesian-testing/)
# Bayesian A/B testing
`bayesian_testing` is a small package for a quick evaluation of A/B (or A/B/C/...) tests using
Bayesian approach.

**Implemented tests:**
- [BinaryDataTest](bayesian_testing/experiments/binary.py)
  - **_Input data_** - binary data (`[0, 1, 0, ...]`)
  - Designed for conversion-like data A/B testing.
- [NormalDataTest](bayesian_testing/experiments/normal.py)
  - **_Input data_** - normal data with unknown variance
  - Designed for normal data A/B testing.
- [DeltaLognormalDataTest](bayesian_testing/experiments/delta_lognormal.py)
  - **_Input data_** - lognormal data with zeros
  - Designed for revenue-like data A/B testing.
- [DeltaNormalDataTest](bayesian_testing/experiments/delta_normal.py)
  - **_Input data_** - normal data with zeros
  - Designed for profit-like data A/B testing.
- [DiscreteDataTest](bayesian_testing/experiments/discrete.py)
  - **_Input data_** - categorical data with numerical categories
  - Designed for discrete data A/B testing (e.g. dice rolls, star ratings, 1-10 ratings, etc.).
- [PoissonDataTest](bayesian_testing/experiments/poisson.py)
  - **_Input data_** - non-negative integers (`[1, 0, 3, ...]`)
  - Designed for poisson data A/B testing.
- [ExponentialDataTest](bayesian_testing/experiments/exponential.py)
  - **_Input data_** - exponential data (non-negative real numbers)
  - Designed for exponential data A/B testing (e.g. session/waiting time, time between events,
etc.).

**Implemented evaluation metrics:**
- `Posterior Mean`
  - Expected value from the posterior distribution for a given variant.
- `Credible Interval`
  - Quantile-based credible intervals based on simulations from posterior distributions (i.e.
empirical).
  - Interval probability (`interval_alpha`) can be set during the evaluation (default value is 95%).
- `Probability of Being Best`
  - Probability that a given variant is best among all variants.
  - By default, `the best` is equivalent to `the greatest` (from a data/metric point of view),
however it is possible to change this by using `min_is_best=True` in the evaluation method
(this can be useful if we try to find the variant with the smallest tested measure).
- `Expected Loss`
  - "Risk" of choosing particular variant over other variants in the test.
  - Measured in same units as a tested measure (e.g. positive rate or average value).

`Credible Interval`, `Probability of Being Best` and `Expected Loss` are calculated using
simulations from posterior distributions (considering given data).


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
- `ExponentialDataTest`

All test classes support two methods to insert the data:
- `add_variant_data` - Adding raw data for a variant as a list of observations (or numpy 1-D array).
- `add_variant_data_agg` - Adding aggregated variant data (this can be practical for a large data,
as the aggregation can be done already on a database level).

Both methods for adding data allow specification of prior distributions
(see details in respective docstrings). Default prior setup should be sufficient for most of the
cases (e.g. cases with unknown priors or large amounts of data).

To get the results of the test, simply call the method `evaluate`.

Probability of being best, expected loss and credible intervals are approximated using simulations,
hence the `evaluate` method can return slightly different values for different runs. To stabilize
it, you can  set the `sim_count` parameter of the `evaluate` to a higher value (default value is
20K), or even use the `seed` parameter to fix it completely.

### BinaryDataTest
Class for a Bayesian A/B test for the binary-like data (e.g. conversions, successes, etc.).

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
results
# print(pd.DataFrame(results).set_index('variant').T.to_markdown(tablefmt="grid"))
```

    +-------------------+-----------+-------------+-------------+
    |                   | A         | B           | C           |
    +===================+===========+=============+=============+
    | totals            | 1500      | 1200        | 1000        |
    +-------------------+-----------+-------------+-------------+
    | positives         | 80        | 80          | 50          |
    +-------------------+-----------+-------------+-------------+
    | positive_rate     | 0.05333   | 0.06667     | 0.05        |
    +-------------------+-----------+-------------+-------------+
    | posterior_mean    | 0.05363   | 0.06703     | 0.05045     |
    +-------------------+-----------+-------------+-------------+
    | credible_interval | [0.04284, | [0.0535309, | [0.0379814, |
    |                   | 0.065501] | 0.0816476]  | 0.0648625]  |
    +-------------------+-----------+-------------+-------------+
    | prob_being_best   | 0.06485   | 0.89295     | 0.0422      |
    +-------------------+-----------+-------------+-------------+
    | expected_loss     | 0.0139248 | 0.0004693   | 0.0170767   |
    +-------------------+-----------+-------------+-------------+

### NormalDataTest
Class for a Bayesian A/B test for the normal data.

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
results = test.evaluate(sim_count=20000, seed=52, min_is_best=False, interval_alpha=0.99)
results
# print(pd.DataFrame(results).set_index('variant').T.to_markdown(tablefmt="grid"))
```

    +-------------------+-------------+-------------+-------------+
    |                   | A           | B           | C           |
    +===================+=============+=============+=============+
    | totals            | 1000        | 800         | 500         |
    +-------------------+-------------+-------------+-------------+
    | sum_values        | 7294.67901  | 5685.86168  | 3736.91581  |
    +-------------------+-------------+-------------+-------------+
    | avg_values        | 7.29468     | 7.10733     | 7.47383     |
    +-------------------+-------------+-------------+-------------+
    | posterior_mean    | 7.29462     | 7.10725     | 7.4737      |
    +-------------------+-------------+-------------+-------------+
    | credible_interval | [7.1359436, | [6.9324733, | [7.0240102, |
    |                   | 7.4528369]  | 7.2779293]  | 7.9379341]  |
    +-------------------+-------------+-------------+-------------+
    | prob_being_best   | 0.1707      | 0.00125     | 0.82805     |
    +-------------------+-------------+-------------+-------------+
    | expected_loss     | 0.1968735   | 0.385112    | 0.0169998   |
    +-------------------+-------------+-------------+-------------+

### DeltaLognormalDataTest
Class for a Bayesian A/B test for the delta-lognormal data (log-normal with zeros).
Delta-lognormal data is typical case of revenue per session data where many sessions have 0 revenue
but non-zero values are positive values with possible log-normal distribution.
To handle this data, the calculation is combining binary Bayes model for zero vs non-zero
"conversions" and log-normal model for non-zero values.

**Example:**
```python
import numpy as np
from bayesian_testing.experiments import DeltaLognormalDataTest

test = DeltaLognormalDataTest()

data_a = [7.1, 0.3, 5.9, 0, 1.3, 0.3, 0, 1.2, 0, 3.6, 0, 1.5,
          2.2, 0, 4.9, 0, 0, 1.1, 0, 0, 7.1, 0, 6.9, 0]
data_b = [4.0, 0, 3.3, 19.3, 18.5, 0, 0, 0, 12.9, 0, 0, 0, 10.2,
          0, 0, 23.1, 0, 3.7, 0, 0, 11.3, 10.0, 0, 18.3, 12.1]

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
results
# print(pd.DataFrame(results).set_index('variant').T.to_markdown(tablefmt="grid"))
```

    +---------------------+-------------+-------------+
    |                     | A           | B           |
    +=====================+=============+=============+
    | totals              | 24          | 25          |
    +---------------------+-------------+-------------+
    | positives           | 13          | 12          |
    +---------------------+-------------+-------------+
    | sum_values          | 43.4        | 146.7       |
    +---------------------+-------------+-------------+
    | avg_values          | 1.80833     | 5.868       |
    +---------------------+-------------+-------------+
    | avg_positive_values | 3.33846     | 12.225      |
    +---------------------+-------------+-------------+
    | posterior_mean      | 2.09766     | 6.19017     |
    +---------------------+-------------+-------------+
    | credible_interval   | [0.9884509, | [3.3746212, |
    |                     | 6.9054963]  | 11.7349253] |
    +---------------------+-------------+-------------+
    | prob_being_best     | 0.04815     | 0.95185     |
    +---------------------+-------------+-------------+
    | expected_loss       | 4.0941101   | 0.1588627   |
    +---------------------+-------------+-------------+

***Note**: Alternatively, `DeltaNormalDataTest` can be used for a case when conversions are not
necessarily positive values.*

### DiscreteDataTest
Class for a Bayesian A/B test for the discrete data with finite number of numerical categories
(states), representing some value.
This test can be used for instance for dice rolls data (when looking for the "best" of multiple
dice) or rating data (e.g. 1-5 stars or 1-10 scale).

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
results = test.evaluate(sim_count=20000, seed=52, min_is_best=False, interval_alpha=0.95)
results
# print(pd.DataFrame(results).set_index('variant').T.to_markdown(tablefmt="grid"))
```

    +-------------------+------------------+------------------+------------------+
    |                   | A                | B                | C                |
    +===================+==================+==================+==================+
    | concentration     | {1: 2.0, 2: 4.0, | {1: 1.0, 2: 6.0, | {1: 1.0, 2: 0.0, |
    |                   | 3: 4.0, 4: 2.0,  | 3: 2.0, 4: 1.0,  | 3: 1.0, 4: 1.0,  |
    |                   | 5: 2.0, 6: 6.0}  | 5: 0.0, 6: 0.0}  | 5: 1.0, 6: 1.0}  |
    +-------------------+------------------+------------------+------------------+
    | average_value     | 3.8              | 2.3              | 3.8              |
    +-------------------+------------------+------------------+------------------+
    | posterior_mean    | 3.73077          | 2.75             | 3.63636          |
    +-------------------+------------------+------------------+------------------+
    | credible_interval | [3.0710797,      | [2.1791584,      | [2.6556465,      |
    |                   | 4.3888021]       | 3.4589178]       | 4.5784839]       |
    +-------------------+------------------+------------------+------------------+
    | prob_being_best   | 0.54685          | 0.008            | 0.44515          |
    +-------------------+------------------+------------------+------------------+
    | expected_loss     | 0.199953         | 1.1826766        | 0.2870247        |
    +-------------------+------------------+------------------+------------------+

### PoissonDataTest
Class for a Bayesian A/B test for the poisson data.

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

# evaluate test (since fewer goals is better, we explicitly set the min_is_best to True)
results = test.evaluate(sim_count=20000, seed=52, min_is_best=True)
results
# print(pd.DataFrame(results).set_index('variant').T.to_markdown(tablefmt="grid"))
```

    +-------------------+-------------+-------------+------------+
    |                   | psg         | city        | bayern     |
    +===================+=============+=============+============+
    | totals            | 15          | 14          | 15         |
    +-------------------+-------------+-------------+------------+
    | sum_values        | 9           | 14          | 13         |
    +-------------------+-------------+-------------+------------+
    | observed_average  | 0.6         | 1.0         | 0.86667    |
    +-------------------+-------------+-------------+------------+
    | posterior_mean    | 0.60265     | 1.13333     | 0.86755    |
    +-------------------+-------------+-------------+------------+
    | credible_interval | [0.2800848, | [0.6562029, | [0.465913, |
    |                   | 1.0570327]  | 1.7265045]  | 1.3964389] |
    +-------------------+-------------+-------------+------------+
    | prob_being_best   | 0.78175     | 0.0344      | 0.18385    |
    +-------------------+-------------+-------------+------------+
    | expected_loss     | 0.0369998   | 0.5620553   | 0.3003345  |
    +-------------------+-------------+-------------+------------+

_note: Since we set `min_is_best=True` (because received goals are "bad"), probability and loss are
in a favor of variants with lower posterior means._

### ExponentialDataTest
Class for a Bayesian A/B test for the exponential data.

**Example:**
```python
import numpy as np
from bayesian_testing.experiments import ExponentialDataTest

# waiting times for 3 different variants, each with many observations,
# generated using exponential distributions with defined scales (expected values)
waiting_times_a = np.random.exponential(scale=10, size=200)
waiting_times_b = np.random.exponential(scale=11, size=210)
waiting_times_c = np.random.exponential(scale=11, size=220)

# initialize a test:
test = ExponentialDataTest()
# adding variants using the observation data:
test.add_variant_data('A', waiting_times_a)
test.add_variant_data('B', waiting_times_b)
test.add_variant_data('C', waiting_times_c)

# alternatively, add variants using aggregated data:
# test.add_variant_data_agg('A', len(waiting_times_a), sum(waiting_times_a))

# evaluate test (since a lower waiting time is better, we set the min_is_best to True)
results = test.evaluate(sim_count=20000, min_is_best=True)
results
# print(pd.DataFrame(results).set_index('variant').T.to_markdown(tablefmt="grid"))
```

    +-------------------+-------------+-------------+-------------+
    |                   | A           | B           | C           |
    +===================+=============+=============+=============+
    | totals            | 200         | 210         | 220         |
    +-------------------+-------------+-------------+-------------+
    | sum_values        | 1827.81709  | 2217.46016  | 2160.73134  |
    +-------------------+-------------+-------------+-------------+
    | observed_average  | 9.13909     | 10.55933    | 9.82151     |
    +-------------------+-------------+-------------+-------------+
    | posterior_mean    | 9.13502     | 10.55478    | 9.8175      |
    +-------------------+-------------+-------------+-------------+
    | credible_interval | [7.994178,  | [9.2543372, | [8.6184821, |
    |                   | 10.5410967] | 12.1527256] | 11.2566538] |
    +-------------------+-------------+-------------+-------------+
    | prob_being_best   | 0.7456      | 0.0405      | 0.2139      |
    +-------------------+-------------+-------------+-------------+
    | expected_loss     | 0.1428729   | 1.5674747   | 0.8230728   |
    +-------------------+-------------+-------------+-------------+

## Development
To set up a development environment, use [Poetry](https://python-poetry.org/) and [pre-commit](https://pre-commit.com):
```console
pip install poetry
poetry install
poetry run pre-commit install
```

## To be implemented

Additional metrics:
- `Potential Value Remaining`

## References
- `bayesian_testing` package itself depends only on [numpy](https://numpy.org) package.
- Work on this package (including default priors selection) was inspired mainly by a Coursera
course [Bayesian Statistics: From Concept to Data Analysis](https://www.coursera.org/learn/bayesian-statistics).
