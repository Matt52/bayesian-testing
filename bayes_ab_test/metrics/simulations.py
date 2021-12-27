from numbers import Number
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray

from bayes_ab_test.utilities import get_logger

logger = get_logger("bayes_ab_test")


def validate_bernoulli_input(totals: List[int], positives: List[int]) -> None:
    """
    Simple validation for pbb_bernoulli_agg inputs.
    """
    if len(totals) != len(positives):
        msg = (
            f"Totals ({totals}) and positives ({positives}) needs to have same length!"
        )
        logger.error(msg)
        raise ValueError(msg)


def beta_posteriors_all(
    totals: List[int],
    positives: List[int],
    sim_count: int,
    a_priors_beta: List[Union[float, int]],
    b_priors_beta: List[Union[float, int]],
    seed: int = None,
) -> ndarray:
    """
    Draw from beta posterior distributions for all variants at once.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    positives : List of numbers of ones (e.g. number of conversions) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    seed : Random seed.

    Returns
    -------
    beta_samples : List of lists of beta distribution samples for all variants.
    """
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random
    beta_samples = np.array(
        [
            rng.beta(
                positives[i] + a_priors_beta[i],
                totals[i] - positives[i] + b_priors_beta[i],
                sim_count,
            )
            for i in range(len(totals))
        ]
    )
    return beta_samples


def pbb_bernoulli_agg(
    totals: List[int],
    positives: List[int],
    a_priors_beta: List[Number] = None,
    b_priors_beta: List[Number] = None,
    sim_count: int = 20000,
    seed: int = None,
) -> List[float]:
    """
    Method estimating probabilities of being best for beta-bernoulli aggregated data per variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    positives : List of numbers of ones (e.g. number of conversions) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    seed : Random seed.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    validate_bernoulli_input(totals, positives)

    if len(totals) == 0:
        return []

    # Default prior for all variants is Beta(0.5, 0.5) which is non-information prior.
    if not a_priors_beta:
        a_priors_beta = [0.5] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [0.5] * len(totals)

    beta_samples = beta_posteriors_all(
        totals, positives, sim_count, a_priors_beta, b_priors_beta, seed
    )

    max_values = np.argmax(beta_samples, axis=0)
    unique, counts = np.unique(max_values, return_counts=True)
    occurrences = dict(zip(unique, counts))

    res = []
    for i in range(len(totals)):
        res.append(round(occurrences.get(i, 0) / sim_count, 7))

    return res


def normal_posteriors(
    total: int,
    sums: float,
    sums_2: float,
    sim_count: int = 20000,
    prior_m: Union[float, int] = 1,
    prior_a: Union[float, int] = 0,
    prior_b: Union[float, int] = 0,
    prior_w: Union[float, int] = 0.01,
    seed: int = None,
) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
    """
    Drawing mus and sigmas from posterior normal distribution considering given aggregated data.

    Parameters
    ----------
    total : Number of data observations from normal data.
    sums : Sum of original data.
    sums_2 : Sum of squares of original data.
    sim_count : Number of simulations.
    prior_m : Prior mean.
    prior_a : Prior alpha from inverse gamma dist. for unknown variance of original data.
        In theory a > 0, but as we always have at least one observation, we can start at 0.
    prior_b : Prior beta from inverse gamma dist. for unknown variance of original data.
        In theory b > 0, but as we always have at least one observation, we can start at 0.
    prior_w : Prior effective sample size.
    seed : Random seed.

    Returns
    -------
    mu_post : List of size sim_count with mus drawn from normal distribution.
    sig_2_post : List of size sim_count with mus drawn from normal distribution.
    """
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random

    x_bar = sums / total
    a_post = prior_a + (total / 2)
    b_post = (
        prior_b
        + (1 / 2) * (sums_2 - 2 * sums * x_bar + total * (x_bar ** 2))
        + ((total * prior_w) / (2 * (total + prior_w))) * ((x_bar - prior_m) ** 2)
    )

    # here it has to be 1/b as it is a scale, and not a rate
    sig_2_post = 1 / rng.gamma(a_post, 1 / b_post, sim_count)

    m_post = (total * x_bar + prior_w * prior_m) / (total + prior_w)

    mu_post = rng.normal(m_post, np.sqrt(sig_2_post / (total + prior_w)))

    return mu_post, sig_2_post


def lognormal_posteriors(
    total: int,
    sum_logs: float,
    sum_logs_2: float,
    sim_count: int = 20000,
    prior_m: Union[float, int] = 1,
    prior_a: Union[float, int] = 0,
    prior_b: Union[float, int] = 0,
    prior_w: Union[float, int] = 0.01,
    seed: int = None,
) -> List[float]:
    """
    Drawing from posterior lognormal distribution using logarithms of original (lognormal) data
    (logarithms of lognormal data are normal). Input data is in aggregated form.

    Parameters
    ----------
    total : Number of lognormal data observations.
        Could be number of conversions in session data.
    sum_logs : Sum of logarithms of original data.
    sum_logs_2 : Sum of logarithms squared of original data.
    sim_count : Number of simulations.
    prior_m : Prior mean of logarithms of original data.
    prior_a : Prior alpha from inverse gamma dist. for unknown variance of logarithms
        of original data. In theory a > 0, but as we always have at least one observation,
        we can start at 0.
    prior_b : Prior beta from inverse gamma dist. for unknown variance of logarithms
        of original data. In theory b > 0, but as we always have at least one observation,
        we can start at 0.
    prior_w : Prior effective sample size.
    seed : Random seed.

    Returns
    -------
    res : List of sim_count numbers drawn from lognormal distribution.
    """
    if total <= 0:
        return list(np.zeros(sim_count))

    # normal posterior for aggregated data of logarithms of original data
    normal_mu_post, normal_sig_2_post = normal_posteriors(
        total, sum_logs, sum_logs_2, sim_count, prior_m, prior_a, prior_b, prior_w, seed
    )

    # final simulated lognormal means using simulated normal means and sigmas
    res = np.exp(normal_mu_post + (normal_sig_2_post / 2))

    return res


def pbb_delta_lognormal_agg(
    totals: List[int],
    non_zeros: List[int],
    sum_logs: List[float],
    sum_logs_2: List[float],
    sim_count: int = 20000,
    a_priors_beta: List[Number] = None,
    b_priors_beta: List[Number] = None,
    m_priors: List[Number] = None,
    a_priors_ig: List[Number] = None,
    b_priors_ig: List[Number] = None,
    w_priors: List[Number] = None,
    seed: int = None,
) -> List[float]:
    """
    Method estimating probabilities of being best for delta-lognormal aggregated data per variant.
    For that reason the method works with both totals and non_zeros.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    non_zeros : List of numbers of non-zeros (e.g. number of conversions) for each variant.
    sum_logs : List of sum of logarithms of original data for each variant.
    sum_logs_2 : List of sum of logarithms squared of original data for each variant.
    sim_count : Number of simulations.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    m_priors : List of prior means for logarithms of non-zero data for each variant.
    a_priors_ig : List of prior alphas from inverse gamma dist approximating variance of logarithms.
    b_priors_ig : List of prior betas from inverse gamma dist approximating variance of logarithms.
    w_priors : List of prior effective sample sizes for each variant.
    seed : Random seed.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    if len(totals) == 0:
        return []
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

    if max(non_zeros) <= 0:
        # if only zeros in all variants
        return list(np.full(len(totals), round(1 / len(totals), 7)))
    else:
        beta_samples = beta_posteriors_all(
            totals, non_zeros, sim_count, a_priors_beta, b_priors_beta, seed
        )

        lognormal_samples = np.array(
            [
                lognormal_posteriors(
                    non_zeros[i],
                    sum_logs[i],
                    sum_logs_2[i],
                    sim_count,
                    m_priors[i],
                    a_priors_ig[i],
                    b_priors_ig[i],
                    w_priors[i],
                    seed,
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
