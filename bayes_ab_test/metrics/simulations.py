from typing import List
from numbers import Number
import numpy as np
from bayes_ab_test.utilities import get_logger

logger = get_logger("bayes_ab_test")


def validate_bernoulli_input(totals: List[int], successes: List[int]) -> None:
    """
    Simple validation for pbb_bernoulli_agg inputs.
    """
    if len(totals) != len(successes):
        msg = f"Totals ({totals}) and successes ({successes}) needs to have same length!"
        logger.error(msg)
        raise ValueError(msg)


def beta_posteriors_all(
    totals: List[int],
    successes: List[int],
    sim_count: int,
    a_priors_beta: List[Number],
    b_priors_beta: List[Number],
) -> List[List[float]]:
    """
    Draw from beta posterior distributions for all variants at once.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    successes : List of numbers of successes (e.g. number of conversions) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.

    Returns
    -------

    """
    beta_samples = np.array(
        [
            np.random.beta(
                successes[i] + a_priors_beta[i],
                totals[i] - successes[i] + b_priors_beta[i],
                sim_count,
            )
            for i in range(len(totals))
        ]
    )
    return beta_samples


def pbb_bernoulli_agg(
    totals: List[int],
    successes: List[int],
    a_priors_beta: List[Number] = None,
    b_priors_beta: List[Number] = None,
    sim_count: int = 20000,
) -> List[float]:
    """
    Method estimating probabilities of being best for beta-bernoulli aggregated data per variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    successes : List of numbers of successes (e.g. number of conversions) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    validate_bernoulli_input(totals, successes)

    # Default prior for all variants is Beta(0.5, 0.5) which is non-information prior.
    if not a_priors_beta:
        a_priors_beta = [0.5] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [0.5] * len(totals)

    beta_samples = beta_posteriors_all(totals, successes, sim_count, a_priors_beta, b_priors_beta)

    max_values = np.argmax(beta_samples, axis=0)
    unique, counts = np.unique(max_values, return_counts=True)
    occurrences = dict(zip(unique, counts))

    res = []
    for i in range(len(totals)):
        res.append(round(occurrences.get(i, 0) / sim_count, 7))

    return res


def lognormal_posteriors(
    totals: int,
    sum_logs: float,
    sum_logs_2: float,
    sim_count: int = 20000,
    prior_m: Number = 1,
    prior_a: Number = 0,
    prior_b: Number = 0,
    prior_w: Number = 0.01,
) -> List[float]:
    """
    Drawing from lognormal distribution using logarithms of original (lognormal) data
    (logarithms of lognormal data are normal).

    Parameters
    ----------
    totals : Number data observations from lognormal data.
        Could be number of conversions in session data.
    sum_logs : Sum of logarithms of original data.
    sum_logs_2 : Sum of logarithms squared of original data.
    sim_count : Number of simulations.
    prior_m : Prior mean of logarithms of original data.
    prior_a : Prior alpha from inverse gamma dist approximating variance of logarithms of original data.
        In theory a > 0, but as we always have at least one observation, we can start at 0.
    prior_b : Prior beta from inverse gamma dist approximating variance of logarithms of original data.
        In theory b > 0, but as we always have at least one observation, we can start at 0.
    prior_w : Effective sample size.

    Returns
    -------
    res : List of sim_count numbers drawn from lognormal distribution.
    """
    if totals <= 0:
        return np.zeros(sim_count)
    else:
        x_bar = sum_logs / totals
        a_post = prior_a + (totals / 2)
        b_post = (
            prior_b
            + (1 / 2) * (sum_logs_2 - 2 * sum_logs * x_bar + totals * (x_bar ** 2))
            + ((totals * prior_w) / (2 * (totals + prior_w))) * ((x_bar - prior_m) ** 2)
        )

        # here it has to be 1/b as it is a scale, and not a rate
        sig_2 = 1 / np.random.gamma(a_post, 1 / b_post, sim_count)

        m_post = (totals * x_bar + prior_w * prior_m) / (totals + prior_w)
        sig_2_post = sig_2 / (totals + prior_w)

        normal_post = np.random.normal(m_post, np.sqrt(sig_2_post))

        # final simulated lognormal means using simulated normal means
        res = np.exp(normal_post + (sig_2 / 2))

        return res


def pbb_lognormal_agg(
    totals: List[int],
    successes: List[int],
    sum_logs: List[float],
    sum_logs_2: List[float],
    sim_count: int = 20000,
    a_priors_beta: List[Number] = None,
    b_priors_beta: List[Number] = None,
    m_priors: List[Number] = None,
    a_priors_ig: List[Number] = None,
    b_priors_ig: List[Number] = None,
    w_priors: List[Number] = None,
) -> List[float]:
    """
    Method estimating probabilities of being best for lognormal aggregated data per variant.
    For convenience, this method allows working with underling data containing also zeros
    which is more practical as revenue-like data can contain many cases with no-revenue
    (e.g. revenue per online shop session where most of the sessions are without any order).
    For that reason the method works with both totals and successes.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    successes : List of numbers of successes (e.g. number of conversions) for each variant.
    sum_logs : List of sum of logarithms of original data for each variant.
    sum_logs_2 : List of sum of logarithms squared of original data for each variant.
    sim_count : Number of simulations.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    m_priors : List of prior means of logarithms of original data for each variant.
    a_priors_ig : List of prior alphas from inverse gamma dist approximating variance of logarithms.
    b_priors_ig : List of prior betas from inverse gamma dist approximating variance of logarithms.
    w_priors : List of effective sample sizes for each variant.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """

    # Same default priors for all variants if they are not provided.
    if not a_priors_beta:
        a_priors_beta = [0.5] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [0.5] * len(totals)
    if not m_priors:
        m_priors = [1] * len(totals)
    if not a_priors_ig:
        a_priors_ig = [0] * len(totals)
    if not b_priors_ig:
        b_priors_ig = [0] * len(totals)
    if not w_priors:
        w_priors = [0.01] * len(totals)

    if max(successes) <= 0:
        # if no success
        return np.full(len(totals), 1 / len(totals))
    else:
        beta_samples = beta_posteriors_all(
            totals, successes, sim_count, a_priors_beta, b_priors_beta
        )

        lognormal_samples = np.array(
            [
                lognormal_posteriors(
                    successes[i],
                    sum_logs[i],
                    sum_logs_2[i],
                    sim_count,
                    m_priors[i],
                    a_priors_ig[i],
                    b_priors_ig[i],
                    w_priors[i],
                )
                for i in range(len(totals))
            ]
        )

        combined_samples = beta_samples * lognormal_samples

        max_values = np.argmax(combined_samples, axis=0)
        unique, counts = np.unique(max_values, return_counts=True)
        occurrences = dict(zip(unique, counts))

        res = []
        for i in range(len(totals)):
            res.append(round(occurrences.get(i, 0) / sim_count, 7))

        return res
