from numbers import Number
from typing import List, Union

import numpy as np

from bayesian_testing.metrics.posteriors import (
    beta_posteriors_all,
    lognormal_posteriors,
    normal_posteriors,
)
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


def validate_bernoulli_input(totals: List[int], positives: List[int]) -> None:
    """
    Simple validation for pbb_bernoulli_agg inputs.
    """
    if len(totals) != len(positives):
        msg = f"Totals ({totals}) and positives ({positives}) needs to have same length!"
        logger.error(msg)
        raise ValueError(msg)


def estimate_probabilities(data: Union[List[List[Number]], np.ndarray]) -> List[float]:
    """
    Estimate probabilities for variants considering simulated data from respective posteriors.

    Parameters
    ----------
    data : List of simulated data for each variant.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    max_values = np.argmax(data, axis=0)
    unique, counts = np.unique(max_values, return_counts=True)
    occurrences = dict(zip(unique, counts))
    sim_count = len(data[0])
    res = []
    for i in range(len(data)):
        res.append(round(occurrences.get(i, 0) / sim_count, 7))
    return res


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

    res = estimate_probabilities(beta_samples)

    return res


def pbb_normal_agg(
    totals: List[int],
    sums: List[float],
    sums_2: List[float],
    sim_count: int = 20000,
    m_priors: List[Number] = None,
    a_priors_ig: List[Number] = None,
    b_priors_ig: List[Number] = None,
    w_priors: List[Number] = None,
    seed: int = None,
) -> List[float]:
    """
    Method estimating probabilities of being best for normal aggregated data per variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations for each variant.
    sums : List of sum of original data for each variant.
    sums_2 : List of sum of squares of original data for each variant.
    sim_count : Number of simulations.
    m_priors : List of prior means for each variant.
    a_priors_ig : List of prior alphas from inverse gamma dist approximating variance.
    b_priors_ig : List of prior betas from inverse gamma dist approximating variance.
    w_priors : List of prior effective sample sizes for each variant.
    seed : Random seed.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    if len(totals) == 0:
        return []
    # Same default priors for all variants if they are not provided.
    if not m_priors:
        m_priors = [1] * len(totals)
    if not a_priors_ig:
        a_priors_ig = [0] * len(totals)
    if not b_priors_ig:
        b_priors_ig = [0] * len(totals)
    if not w_priors:
        w_priors = [0.01] * len(totals)

    # we will need different generators for each call of normal_posteriors
    # (so they are not perfectly correlated)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(totals))

    normal_samples = np.array(
        [
            normal_posteriors(
                totals[i],
                sums[i],
                sums_2[i],
                sim_count,
                m_priors[i],
                a_priors_ig[i],
                b_priors_ig[i],
                w_priors[i],
                child_seeds[i],
            )[0]
            for i in range(len(totals))
        ]
    )

    res = estimate_probabilities(normal_samples)

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
        # we will need different generators for each call of lognormal_posteriors
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(len(totals) + 1)

        beta_samples = beta_posteriors_all(
            totals, non_zeros, sim_count, a_priors_beta, b_priors_beta, child_seeds[0]
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
                    child_seeds[1 + i],
                )
                for i in range(len(totals))
            ]
        )

        combined_samples = beta_samples * lognormal_samples

        res = estimate_probabilities(combined_samples)

        return res
