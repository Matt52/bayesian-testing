from typing import List
from numbers import Number
from bayes_ab_test.metrics import pbb_bernoulli_agg
from bayes_ab_test.utilities.common import check_list_lengths
from bayes_ab_test.utilities import get_logger

logger = get_logger("bayes_ab_test")


class ConversionTest:
    def __init__(
        self,
        variant_names: List[str] = [],
        totals: List[int] = [],
        successes: List[int] = [],
        a_priors_beta: List[Number] = [],
        b_priors_beta: List[Number] = [],
        sim_count: int = 20000,
    ) -> None:
        """
        Initialize ConversionTest.

        Parameters
        ----------
        variant_names : List of variant names as strings.
        totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
        successes : List of numbers of successes (e.g. number of conversions) for each variant.
        a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
        b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
        """
        self.variant_names = variant_names.copy()
        self.totals = totals.copy()
        self.successes = successes.copy()
        self.a_priors_beta = a_priors_beta.copy()
        self.b_priors_beta = b_priors_beta.copy()
        self.sim_count = sim_count
        self.validate_init_input()

    def validate_init_input(self) -> None:
        """
        Basic initialization validation. Applying default priors in case they were not specified.
        """
        if not self.a_priors_beta and not self.b_priors_beta:
            # default prior is non-information prior Beta(0.5, 0.5) for all variants.
            self.a_priors_beta = [0.5] * len(self.totals)
            self.b_priors_beta = [0.5] * len(self.totals)
        elif not self.a_priors_beta or not self.b_priors_beta:
            msg = "As one of [a_priors_beta, b_priors_beta] is empty, both will be set to 0.5 for all variants."
            logger.warning(msg)
            self.a_priors_beta = [0.5] * len(self.totals)
            self.b_priors_beta = [0.5] * len(self.totals)

        check_list_lengths(
            [
                self.variant_names,
                self.totals,
                self.successes,
                self.a_priors_beta,
                self.b_priors_beta,
            ]
        )

        for v in self.variant_names:
            if not isinstance(v, str):
                raise ValueError("All variant names need to be strings.")

        if len(set(self.variant_names)) != len(self.variant_names):
            raise ValueError("All variant names need to be unique.")

    @property
    def probabs_of_being_best(self):
        pbbs = pbb_bernoulli_agg(
            self.totals, self.successes, self.a_priors_beta, self.b_priors_beta, self.sim_count
        )
        return pbbs

    def evaluate(self):
        keys = ["variant", "totals", "successes", "conv. rate", "prob. being best"]
        conv_rates = [round(i[0] / i[1], 5) for i in zip(self.successes, self.totals)]
        data = [
            self.variant_names,
            self.totals,
            self.successes,
            conv_rates,
            self.probabs_of_being_best,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        return res

    def add_variant(
        self, name: str, data: List[int], a_prior: Number = 0.5, b_prior: Number = 0.5
    ) -> None:
        """
        Add variant to test class using raw conversion data.

        Parameters
        ----------
        name : Variant name.
        data : List of conversion data containing zeros (non-conversion) and ones (conversions).
        a_prior : Prior alpha parameter for Beta distributions.
            Default value 0.5 is based on non-information prior Beta(0.5, 0.5).
        b_prior : Prior beta parameter for Beta distributions.
            Default value 0.5 is based on non-information prior Beta(0.5, 0.5).
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if name in self.variant_names:
            raise ValueError(f"Variant {name} already exists.")
        if a_prior < 0 or b_prior < 0:
            raise ValueError("Both [a_prior, b_prior] have to be non-negative.")
        if len(data) == 0:
            raise ValueError("Data of new variant needs to have some observations.")
        if not min([i in [0, 1] for i in data]):
            raise ValueError("Input data needs to be a list with zeros and ones.")

        self.variant_names.append(name)
        self.totals.append(len(data))
        self.successes.append(sum(data))
        self.a_priors_beta.append(a_prior)
        self.b_priors_beta.append(b_prior)
