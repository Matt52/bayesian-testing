from numbers import Number
from typing import List, Tuple

import numpy as np

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import eval_delta_lognormal_agg
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


class DeltaLognormalDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for Delta-LogNormal data (Log-Normal with possible zeros).
    Delta-lognormal data is typical case of revenue/session data where many
    sessions are with 0 revenue (meaning non-conversions).
    To handle this data, the evaluation methods are combining binary bayes model for
    zero vs non-zero "conversion" and log-normal model for non-zero values.

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self) -> None:
        """
        Initialize DeltaLognormalDataTest class.
        """
        super().__init__()

    @property
    def totals(self):
        return [self.data[k]["totals"] for k in self.data]

    @property
    def positives(self):
        return [self.data[k]["positives"] for k in self.data]

    @property
    def sum_values(self):
        return [self.data[k]["sum_values"] for k in self.data]

    @property
    def sum_logs(self):
        return [self.data[k]["sum_logs"] for k in self.data]

    @property
    def sum_logs_2(self):
        return [self.data[k]["sum_logs_2"] for k in self.data]

    @property
    def a_priors_beta(self):
        return [self.data[k]["a_prior_beta"] for k in self.data]

    @property
    def b_priors_beta(self):
        return [self.data[k]["b_prior_beta"] for k in self.data]

    @property
    def m_priors(self):
        return [self.data[k]["m_prior"] for k in self.data]

    @property
    def a_priors_ig(self):
        return [self.data[k]["a_prior_ig"] for k in self.data]

    @property
    def b_priors_ig(self):
        return [self.data[k]["b_prior_ig"] for k in self.data]

    @property
    def w_priors(self):
        return [self.data[k]["w_prior"] for k in self.data]

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
        pbbs, loss, intervals, hdis = eval_delta_lognormal_agg(
            self.totals,
            self.positives,
            self.sum_logs,
            self.sum_logs_2,
            sim_count=sim_count,
            a_priors_beta=self.a_priors_beta,
            b_priors_beta=self.b_priors_beta,
            m_priors=self.m_priors,
            a_priors_ig=self.a_priors_ig,
            b_priors_ig=self.b_priors_ig,
            w_priors=self.w_priors,
            seed=seed,
            min_is_best=min_is_best,
            interval_alpha=interval_alpha,
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))
        res_intervals = dict(zip(self.variant_names, intervals))
        res_hdis= dict(zip(self.variant_names, hdis))

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
            "positives",
            "sum_values",
            "avg_values",
            "avg_positive_values",
            "posterior_mean",
            "credible_interval",
            "high_density_interval",
            "prob_being_best",
            "expected_loss",
            "posteriors"
        ]
        avg_values = [round(i[0] / i[1], 5) for i in zip(self.sum_values, self.totals)]
        avg_pos_values = [round(i[0] / i[1], 5) for i in zip(self.sum_values, self.positives)]
        a_posterior_ig = [i[0] + (i[1] / 2) for i in zip(self.a_priors_ig, self.positives)]
        x_ig = [i[0] / i[1] for i in zip(self.sum_logs, self.positives)]
        b_posterior_ig = [
            (
                i[6]
                + (1 / 2) * (i[1] - 2 * i[0] * i[3] + i[2] * (i[3] ** 2))
                + ((i[2] * i[5]) / (2 * (i[2] + i[5]))) * ((i[3] - i[4]) ** 2)
            )
            for i in zip(
                self.sum_logs,
                self.sum_logs_2,
                self.positives,
                x_ig,
                self.m_priors,
                self.w_priors,
                self.b_priors_ig,
            )
        ]
        posterior_mean = [
            round(
                np.exp(((i[0] + i[3] * i[4]) / (i[1] + i[4])) + i[8] / (2 * i[7]))
                * ((i[5] + i[1]) / (i[6] + i[2])),
                5,
            )
            for i in zip(
                self.sum_logs,
                self.positives,
                self.totals,
                self.m_priors,
                self.w_priors,
                self.a_priors_beta,
                self.b_priors_beta,
                a_posterior_ig,
                b_posterior_ig,
            )
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
            self.positives,
            [round(i, 5) for i in self.sum_values],
            avg_values,
            avg_pos_values,
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
        positives: int,
        sum_values: float,
        sum_logs: float,
        sum_logs_2: float,
        a_prior_beta: Number = 0.5,
        b_prior_beta: Number = 0.5,
        m_prior: Number = 1,
        a_prior_ig: Number = 0,
        b_prior_ig: Number = 0,
        w_prior: Number = 0.01,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using aggregated Delta-LogNormal data.
        This can be convenient as aggregation can be done on database level.

        The goal of default prior setup is to be low information.
        It should be tuned with caution.

        Parameters
        ----------
        name : Variant name.
        totals : Total number of experiment observations (e.g. number of sessions).
        positives : Total number of non-zero values for a given variant.
        sum_values : Sum of non-zero values for a given variant.
        sum_logs : Sum of logarithms of non-zero data values for a given variant.
        sum_logs_2 : Sum of logarithms squrared of non-zero data values for a given variant.
        a_prior_beta : Prior alpha parameter from Beta distribution for conversion part.
        b_prior_beta : Prior beta parameter from Beta distribution for conversion part.
        m_prior : Prior normal mean for logarithms of non-zero data.
        a_prior_ig : Prior alpha from inverse gamma dist. for unknown variance of logarithms.
            In theory a > 0, but as we always have at least one observation, we can start at 0.
        b_prior_ig : Prior beta from inverse gamma dist. for unknown variance of logarithms.
            In theory b > 0, but as we always have at least one observation, we can start at 0.
        w_prior : Prior effective sample sizes for normal distribution of logarithms of data.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if a_prior_beta <= 0 or b_prior_beta <= 0:
            raise ValueError("Both [a_prior_beta, b_prior_beta] have to be positive numbers.")
        if m_prior < 0 or a_prior_ig < 0 or b_prior_ig < 0 or w_prior < 0:
            raise ValueError("All priors of [m, a_ig, b_ig, w] have to be non-negative numbers.")
        if positives == 0:
            raise ValueError("Variant has to have some non-zero (positive) values.")
        if positives < 0:
            raise ValueError("Input variable 'positives' is expected to be a positive integer.")
        if totals < positives:
            raise ValueError("Not possible to have more positives that totals!")

        if name not in self.variant_names:
            self.data[name] = {
                "totals": totals,
                "positives": positives,
                "sum_values": sum_values,
                "sum_logs": sum_logs,
                "sum_logs_2": sum_logs_2,
                "a_prior_beta": a_prior_beta,
                "b_prior_beta": b_prior_beta,
                "m_prior": m_prior,
                "a_prior_ig": a_prior_ig,
                "b_prior_ig": b_prior_ig,
                "w_prior": w_prior,
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {
                "totals": totals,
                "positives": positives,
                "sum_values": sum_values,
                "sum_logs": sum_logs,
                "sum_logs_2": sum_logs_2,
                "a_prior_beta": a_prior_beta,
                "b_prior_beta": b_prior_beta,
                "m_prior": m_prior,
                "a_prior_ig": a_prior_ig,
                "b_prior_ig": b_prior_ig,
                "w_prior": w_prior,
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["totals"] += totals
            self.data[name]["positives"] += positives
            self.data[name]["sum_values"] += sum_values
            self.data[name]["sum_logs"] += sum_logs
            self.data[name]["sum_logs_2"] += sum_logs_2

    def add_variant_data(
        self,
        name: str,
        data: List[Number],
        a_prior_beta: Number = 0.5,
        b_prior_beta: Number = 0.5,
        m_prior: Number = 1,
        a_prior_ig: Number = 0,
        b_prior_ig: Number = 0,
        w_prior: Number = 0.01,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw Delta-LogNormal data.

        The goal of default prior setup is to be low information. It should be tuned with caution.

        Parameters
        ----------
        name : Variant name.
        data : List of delta-lognormal data (e.g. revenues in sessions).
        a_prior_beta : Prior alpha parameter from Beta distribution for conversion part.
        b_prior_beta : Prior beta parameter from Beta distribution for conversion part.
        m_prior : Prior mean for logarithms of non-zero data.
        a_prior_ig : Prior alpha from inverse gamma dist. for unknown variance of logarithms.
            In theory a > 0, but as we always have at least one observation, we can start at 0.
        b_prior_ig : Prior beta from inverse gamma dist. for unknown variance of logarithms.
            In theory b > 0, but as we always have at least one observation, we can start at 0.
        w_prior : Prior effective sample sizes for normal distribution of logarithms of data.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if min(data) < 0:
            raise ValueError("Input data needs to be a list of non-negative numbers.")

        totals = len(data)
        positives = sum(x > 0 for x in data)
        sum_values = sum(data)
        sum_logs = sum([np.log(x) for x in data if x > 0])
        sum_logs_2 = sum([np.square(np.log(x)) for x in data if x > 0])

        self.add_variant_data_agg(
            name,
            totals,
            positives,
            sum_values,
            sum_logs,
            sum_logs_2,
            a_prior_beta,
            b_prior_beta,
            m_prior,
            a_prior_ig,
            b_prior_ig,
            w_prior,
            replace,
        )
