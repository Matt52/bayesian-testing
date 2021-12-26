[![Tests](https://github.com/Matt52/bayes-ab-test/workflows/Tests/badge.svg)](https://github.com/Matt52/bayes-ab-test/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/Matt52/bayes-ab-test/branch/main/graph/badge.svg?token=667K54UO8K)](https://codecov.io/gh/Matt52/bayes-ab-test)
# bayes-ab-test: Bayesian A/B testing
`bayes-ab-test` is a small package for a quick evaluation of A/B (or A/B/C/...) tests using Bayesian approach.

The core evaluation metric of the approach is `Probability of Being Best` which is calculated using simulations
from posterior distribution (considering given data).

The package currently supports these data inputs:
- **binary data** (`[0, 1, 0, ...]`) - convenient for conversion-like A/B testing
- **delta-lognormal data** (`[0, 21.2, 0, ...]`) - convenient for revenue-like A/B testing


## Installation
`bayes-ab-test` can be installed using pip:
```console
pip install bayes-ab-test
```
Alternatively, you can clone the repository and use `poetry` manually:
```console
cd bayes-ab-test
pip install poetry
poetry install
```

## Basic Usage
The primary features are `BinaryDataTest` and `DeltaLognormalDataTest` classes.

In both cases, there are two methods to insert data:
- `add_variant_data` - adding raw data for a variant as a list of numbers (or numpy 1-D array)
- `add_variant_data_agg` - adding aggregated variant data (this can be practical for large data as the aggregation can be done on a database level)

To get the results of the test, simply call method `evaluate`, or `probabs_of_being_best` for returning just the probabilities.

Probabilities of being best are approximated using simulations. Hence `evaluate` can return slightly different
values for different runs. To stabilize it, you can set `sim_count` parameter of `evaluate` to higher value
(default value is 20K), or even use `seed` parameter to fix it completely.


### BinaryDataTest
Class for Bayesian A/B test for binary-like data (e.g. conversions, successes, etc.).

```python
import numpy as np
from bayes_ab_test.experiments import BinaryDataTest


# random 1x1500 array of 0/1 data with 5.2% probability for 1:
data_a = np.random.default_rng(21).binomial(n=1, p=0.052, size=1500)

# random 1x1200 array of 0/1 data with 6.7% probability for 1:
data_b = np.random.default_rng(21).binomial(n=1, p=0.067, size=1200)

# initialize test
test = BinaryDataTest()

# adding variant using raw data (arrays of zeros and ones):
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)

# adding variant using aggregated data (same as raw data with 950 zeros and 50 ones):
test.add_variant_data_agg("C", totals=1000, positives=50)

# evaluate test
test.evaluate()
```

    [{'variant': 'A',
      'totals': 1500,
      'positives': 80,
      'conv_rate': 0.05333,
      'prob_being_best': 0.1198},
     {'variant': 'B',
      'totals': 1200,
      'positives': 76,
      'conv_rate': 0.06333,
      'prob_being_best': 0.8058},
     {'variant': 'C',
      'totals': 1000,
      'positives': 50,
      'conv_rate': 0.05,
      'prob_being_best': 0.0744}]

### DeltaLognormalDataTest
Class for Bayesian A/B test for delta-lognormal data (log-normal with zeros).
Delta-lognormal data is typical case of revenue per session data where many sessions have 0 revenue
but non-zero values are positive numbers with possible log-normal distribution.
To handle this data, the calculation is combining binary bayes model for zero vs non-zero
"conversions" and log-normal model for non-zero values.

```python
import numpy as np
from bayes_ab_test.experiments import DeltaLognormalDataTest

test = DeltaLognormalDataTest()

data_a = [7.1, 0.3, 5.9, 0, 1.3, 0.3, 0, 0, 0, 0, 0, 1.5, 2.2, 0, 4.9, 0, 0, 0, 0, 0]
data_b = [4.0, 0, 3.3, 19.3, 18.5, 0, 0, 0, 12.9, 0, 0, 0, 0, 0, 0, 0, 0, 3.7, 0, 0]

# adding variant using raw data 
test.add_variant_data("A", data_a)

# alternatively, variant can be also added using aggregated data:
test.add_variant_data_agg(
    name="B",
    totals=len(data_b),
    positives=sum(x > 0 for x in data_b),
    sum_values=sum(data_b),
    sum_logs=sum([np.log(x) for x in data_b if x > 0]),
    sum_logs_2=sum([np.square(np.log(x)) for x in data_b if x > 0])
)

test.evaluate(seed=21)
```

    [{'variant': 'A',
      'totals': 20,
      'positives': 8,
      'sum_values': 23.5,
      'avg_values': 1.175,
      'avg_positive_values': 2.9375,
      'prob_being_best': 0.1913},
     {'variant': 'B',
      'totals': 20,
      'positives': 6,
      'sum_values': 61.7,
      'avg_values': 3.085,
      'avg_positive_values': 10.28333,
      'prob_being_best': 0.8087}]

## Development
To set up development environment use [Poetry](https://python-poetry.org/) and [pre-commit](https://pre-commit.com):
```console
pip install poetry
poetry install
pre-commit install
```

## Roadmap

Test classes to be added:
- `PoissonDataTest`
- `ExponentialDataTest`
- `NormalDataTest`

Metrics to be added:
- `Expected Loss`
- `Potential Value Remaining`

## References
- `bayes-ab-test` package itself is dependent only on [numpy](https://numpy.org) package.
- Work on this package (including default priors selection) was inspired mainly by Coursera
course [Bayesian Statistics: From Concept to Data Analysis](https://www.coursera.org/learn/bayesian-statistics).
