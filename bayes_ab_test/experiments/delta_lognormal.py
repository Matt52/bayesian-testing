# import warnings
# from numbers import Number
from typing import List

from bayes_ab_test.metrics import pbb_lognormal_agg
from bayes_ab_test.utilities import get_logger

logger = get_logger("bayes_ab_test")


class DeltaLognormalDataTest:
    def __init__(self) -> None:
        """
        Initialize ConversionTest class.
        """
        self.data = {}

    @property
    def variant_names(self):
        return [k for k in self.data]

    @property
    def totals(self):
        return [self.data[k]["totals"] for k in self.data]

    @property
    def positives(self):
        return [self.data[k]["successes"] for k in self.data]

    @property
    def sum_logs(self):
        return [self.data[k]["sum_logs"] for k in self.data]

    @property
    def sum_logs_2(self):
        return [self.data[k]["sum_logs_2"] for k in self.data]

    @property
    def a_priors_beta(self):
        return [self.data[k]["a_priors_beta"] for k in self.data]

    @property
    def b_priors_beta(self):
        return [self.data[k]["b_priors_beta"] for k in self.data]

    @property
    def m_priors(self):
        return [self.data[k]["m_priors"] for k in self.data]

    @property
    def a_priors_ig(self):
        return [self.data[k]["a_priors_ig"] for k in self.data]

    @property
    def b_priors_ig(self):
        return [self.data[k]["b_priors_ig"] for k in self.data]

    @property
    def w_priors(self):
        return [self.data[k]["w_priors"] for k in self.data]

    def probabs_of_being_best(self, sim_count: int = 20000, seed: int = None) -> dict:
        """
        Calculate probabilities of being best for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        res : Dictionary with probabilities of being best for all variants in experiment.
        """
        pbbs = pbb_lognormal_agg(
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
        )
        res = dict(zip(self.variant_names, pbbs))
        return res

    def evaluate(self, sim_count: int = 20000, seed: int = None) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = [
            "variant",
            "totals",
            "positives",
            "total revenue",
            "avg revenue",
            "avg non-zero revenue",
            "prob. being best",
        ]
        print(keys)
