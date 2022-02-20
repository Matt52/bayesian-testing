from numbers import Number
from typing import List, Union
import numpy as np

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import pbb_numerical_dirichlet_agg
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


class DiscreteDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for finite discrete data (i.e. categorical data
    with numerical categories). As a real world examples we can think of dice rolls,
    1-5 star ratings, 1-10 ratings, etc.

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self, categories: List[Union[float, int]]) -> None:
        """
        Initialize DiscreteDataTest class.
        """
        super().__init__()
        if not self.check_if_numerical(categories):
            raise ValueError("Categories in this test has to be all numbers (int or float).")
        self.categories = categories

    @property
    def concentrations(self):
        return [self.data[k]["concentration"] for k in self.data]

    @property
    def prior_alphas(self):
        return [self.data[k]["prior"] for k in self.data]

    @staticmethod
    def check_if_numerical(values):
        res = True
        for v in values:
            if not isinstance(v, Number):
                res = False
        return res

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
        pbbs = pbb_numerical_dirichlet_agg(
            self.categories, self.concentrations, self.prior_alphas, sim_count, seed
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
        keys = ["variant", "concentration", "average_value", "prob_being_best"]
        pbbs = list(self.probabs_of_being_best(sim_count, seed).values())
        average_values = [
            np.sum(np.multiply(i, self.categories)) / np.sum(i) for i in self.concentrations
        ]
        data = [
            self.variant_names,
            [dict(zip(self.categories, i)) for i in self.concentrations],
            average_values,
            pbbs,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        return res

    def add_variant_data_agg(
        self,
        name: str,
        concentration: List[int],
        prior: List[Union[float, int]] = None,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using aggregated discrete data.
        This can be convenient as aggregation can be done on database level.

        Default prior setup is Dirichlet(1,...,1) which is low information prior
        (we can interpret it as prior 1 observation of each category).

        Parameters
        ----------
        name : Variant name.
        concentration : Total number of experiment observations for each category
            (e.g. number of rolls for each side in a die roll data).
        prior : Prior alpha parameters of Dirichlet distribution.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if not len(self.categories) == len(concentration):
            msg = (
                f"Concentration list has to have same size as number of categories in a test "
                f"{len(concentration)} != {len(self.categories)}."
            )
            raise ValueError(msg)
        if not self.check_if_numerical(concentration):
            raise ValueError("Concentration parameter has to be a list of integer values.")

        if not prior:
            prior = [1] * len(self.categories)

        if name not in self.variant_names:
            self.data[name] = {"concentration": concentration, "prior": prior}
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {"concentration": concentration, "prior": prior}
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["concentration"] = [
                sum(x) for x in zip(self.data[name]["concentration"], concentration)
            ]

    def add_variant_data(
        self,
        name: str,
        data: List[int],
        prior: List[Union[float, int]] = None,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw discrete data.

        Default prior setup is Dirichlet(1,...,1) which is low information prior
        (we can interpret it as prior 1 observation of each category).

        Parameters
        ----------
        name : Variant name.
        data : List of numerical data observations from possible categories.
        prior : Prior alpha parameters of Dirichlet distribution.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not min([i in self.categories for i in data]):
            msg = (
                "Input data needs to be a list of numbers from possible categories: "
                f"{self.categories}."
            )
            raise ValueError(msg)

        counter_dict = dict(zip(self.categories, np.zeros(len(self.categories))))
        for i in data:
            counter_dict[i] += 1
        concentration = [counter_dict[i] for i in self.categories]

        self.add_variant_data_agg(name, concentration, prior, replace)
