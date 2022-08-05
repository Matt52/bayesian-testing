from typing import Tuple
import warnings


class BaseDataTest:
    """
    Base class for Bayesian A/B test.
    """

    def __init__(self) -> None:
        """
        Initialize BaseDataTest class.
        """
        self.data = {}

    @property
    def variant_names(self):
        return [k for k in self.data]

    def eval_simulation(self, sim_count: int = 20000, seed: int = None) -> Tuple[dict, dict]:
        """
        Should be implemented in each individual experiment.
        """
        raise NotImplementedError

    def probabs_of_being_best(self, sim_count: int = 20000, seed: int = None) -> dict:
        """
        Calculate probabilities of being best for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        pbbs : Dictionary with probabilities of being best for all variants in experiment.
        """
        pbbs, loss = self.eval_simulation(sim_count, seed)

        return pbbs

    def expected_loss(self, sim_count: int = 20000, seed: int = None) -> dict:
        """
        Calculate expected loss for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        loss : Dictionary with expected loss for all variants in experiment.
        """
        pbbs, loss = self.eval_simulation(sim_count, seed)

        return loss

    def delete_variant(self, name: str) -> None:
        """
        Delete variant and all its data from experiment.

        Parameters
        ----------
        name : Variant name.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if name not in self.variant_names:
            warnings.warn(f"Nothing to be deleted. Variant {name} is not in experiment.")
        else:
            del self.data[name]
