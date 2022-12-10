from numbers import Number
from typing import List, Tuple, Union
import numpy as np

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import eval_numerical_dirichlet_agg
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


class DiscreteDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for data with finite discrete states (i.e. categorical data
    with numerical categories). As a real world examples we can think of dice rolls,
    1-5 star ratings, 1-10 ratings, etc.

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self, states: List[Union[float, int]]) -> None:
        """
        Initialize DiscreteDataTest class.

        Parameters
        ----------
        states : List of all possible states for a given discrete variable.
        """
        super().__init__()
        if not self.check_if_numerical(states):
            raise ValueError("States in the test have to be numbers (int or float).")
        self.states = states

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

    def eval_simulation(
        self, sim_count: int = 20000, seed: int = None, min_is_best: bool = False
    ) -> Tuple[dict, dict]:
        """
        Calculate probabilities of being best and expected loss for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.
        min_is_best : Option to change "being best" to a minimum. Default is maximum.

        Returns
        -------
        res_pbbs : Dictionary with probabilities of being best for all variants in experiment.
        res_loss : Dictionary with expected loss for all variants in experiment.
        """
        pbbs, loss = eval_numerical_dirichlet_agg(
            self.states, self.concentrations, self.prior_alphas, sim_count, seed, min_is_best
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))

        return res_pbbs, res_loss

    def evaluate(
        self, sim_count: int = 20000, seed: int = None, min_is_best: bool = False
    ) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.
        min_is_best : Option to change "being best" to a minimum. Default is maximum.

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = ["variant", "concentration", "average_value", "prob_being_best", "expected_loss"]
        eval_pbbs, eval_loss = self.eval_simulation(sim_count, seed, min_is_best)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())
        average_values = [
            np.sum(np.multiply(i, self.states)) / np.sum(i) for i in self.concentrations
        ]
        data = [
            self.variant_names,
            [dict(zip(self.states, i)) for i in self.concentrations],
            average_values,
            pbbs,
            loss,
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
        (we can interpret it as prior 1 observation of each state).

        Parameters
        ----------
        name : Variant name.
        concentration : Total number of experiment observations for each state
            (e.g. number of rolls for each side in a die roll data).
        prior : Prior alpha parameters of Dirichlet distribution.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if not len(self.states) == len(concentration):
            msg = (
                f"Concentration list has to have same size as number of states in a test "
                f"{len(concentration)} != {len(self.states)}."
            )
            raise ValueError(msg)
        if not self.check_if_numerical(concentration):
            raise ValueError("Concentration parameter has to be a list of integer values.")

        if not prior:
            prior = [1] * len(self.states)

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
        (we can interpret it as prior 1 observation of each state).

        Parameters
        ----------
        name : Variant name.
        data : List of numerical data observations from possible states.
        prior : Prior alpha parameters of Dirichlet distribution.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not min([i in self.states for i in data]):
            msg = (
                "Input data needs to be a list of numbers from possible states: " f"{self.states}."
            )
            raise ValueError(msg)

        counter_dict = dict(zip(self.states, np.zeros(len(self.states))))
        for i in data:
            counter_dict[i] += 1
        concentration = [counter_dict[i] for i in self.states]

        self.add_variant_data_agg(name, concentration, prior, replace)
