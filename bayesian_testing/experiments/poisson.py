from numbers import Number
from typing import List, Tuple, Union

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import eval_poisson_agg
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


class PoissonDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for Poisson data (i.e. numbers of events, e.g. goals scored).

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self) -> None:
        """
        Initialize BinaryDataTest class.
        """
        super().__init__()

    @property
    def totals(self):
        return [self.data[k]["totals"] for k in self.data]

    @property
    def sum_values(self):
        return [self.data[k]["sum_values"] for k in self.data]

    @property
    def a_priors(self):
        return [self.data[k]["a_prior"] for k in self.data]

    @property
    def b_priors(self):
        return [self.data[k]["b_prior"] for k in self.data]

    def eval_simulation(
        self,
        sim_count: int = 20000,
        seed: int = None,
        min_is_best: bool = False,
        interval_alpha: float = 0.95,
    ) -> Tuple[dict, dict, dict]:
        """
        Calculate probabilities of being best, expected loss and credible intervals for a current
        class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.
        min_is_best : Option to change "being best" to a minimum. Default is maximum.
        interval_alpha : Credible interval probability (value between 0 and 1).

        Returns
        -------
        res_pbbs : Dictionary with probabilities of being best for all variants in experiment.
        res_loss : Dictionary with expected loss for all variants in experiment.
        res_intervals : Dictionary with quantile-based credible intervals for all variants.
        """
        pbbs, loss, intervals, hdis = eval_poisson_agg(
            self.totals,
            self.sum_values,
            self.a_priors,
            self.b_priors,
            sim_count,
            seed,
            min_is_best,
            interval_alpha,
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))
        res_intervals = dict(zip(self.variant_names, intervals))
        res_hdis = dict(zip(self.variant_names, hdis))

        return res_pbbs, res_loss, res_intervals, res_hdis

    def evaluate(
        self,
        sim_count: int = 20000,
        seed: int = None,
        min_is_best: bool = False,
        interval_alpha: float = 0.95,
    ) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.
        min_is_best : Option to change "being best" to a minimum. Default is maximum.
        interval_alpha : Credible interval probability (value between 0 and 1).

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = [
            "variant",
            "totals",
            "sum_values",
            "observed_average",
            "posterior_mean",
            "credible_interval",
            "high_density_interval",
            "prob_being_best",
            "expected_loss",
        ]
        observed_average = [round(i[0] / i[1], 5) for i in zip(self.sum_values, self.totals)]
        posterior_mean = [
            round((i[2] + i[0]) / (i[3] + i[1]), 5)
            for i in zip(self.sum_values, self.totals, self.a_priors, self.b_priors)
        ]
        eval_pbbs, eval_loss, eval_intervals, eval_hdis = self.eval_simulation(
            sim_count, seed, min_is_best, interval_alpha
        )
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())
        intervals = list(eval_intervals.values())
        hdis = list(eval_hdis.values())

        data = [
            self.variant_names,
            self.totals,
            self.sum_values,
            observed_average,
            posterior_mean,
            intervals,
            hdis,
            pbbs,
            loss,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        return res

    def add_variant_data_agg(
        self,
        name: str,
        totals: int,
        sum_values: Union[float, int],
        a_prior: Number = 0.1,
        b_prior: Number = 0.1,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using aggregated Poisson data.
        This can be convenient as aggregation can be done on database level.

        Default prior setup is set for Gamma(0.1, 0.1) which is on purpose very vague prior.

        Parameters
        ----------
        name : Variant name.
        totals : Total number of experiment observations (e.g. number of matches).
        sum_values : Sum of values for a given variant (e.g. total number of goals).
        a_prior : Prior alpha parameter of a Gamma distribution (conjugate prior).
            Default value 0.1 is on purpose to be vague (lower information).
        b_prior : Prior beta parameter (rate) of a Gamma distribution (conjugate prior).
            Default value 0.1 is on purpose to be vague (lower information).
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        if totals <= 0:
            raise ValueError("Input variable 'totals' is expected to be positive integer.")
        if sum_values < 0:
            raise ValueError("Input variable 'sum_values' is expected to be non-negative number.")

        if name not in self.variant_names:
            self.data[name] = {
                "totals": totals,
                "sum_values": sum_values,
                "a_prior": a_prior,
                "b_prior": b_prior,
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {
                "totals": totals,
                "sum_values": sum_values,
                "a_prior": a_prior,
                "b_prior": b_prior,
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["totals"] += totals
            self.data[name]["sum_values"] += sum_values

    def add_variant_data(
        self,
        name: str,
        data: List[int],
        a_prior: Number = 0.1,
        b_prior: Number = 0.1,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw Poisson data.

        Default prior setup is set for Gamma(0.1, 0.1) which is non-information prior.

        Parameters
        ----------
        name : Variant name.
        data : List of Poisson data.
        a_prior : Prior alpha parameter of a Gamma distribution (conjugate prior).
            Default value 0.1 is on purpose to be vague (lower information).
        b_prior : Prior beta parameter (rate) of a Gamma distribution (conjugate prior).
            Default value 0.1 is on purpose to be vague (lower information).
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not min([i >= 0 for i in data]):
            raise ValueError("Input data needs to be a list of non-negative integers.")

        totals = len(data)
        sum_values = sum(data)

        self.add_variant_data_agg(name, totals, sum_values, a_prior, b_prior, replace)
