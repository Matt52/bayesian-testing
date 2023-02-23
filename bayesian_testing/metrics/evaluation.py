from numbers import Number
from typing import List, Tuple, Union

import numpy as np

from bayesian_testing.metrics.posteriors import (
    beta_posteriors_all,
    lognormal_posteriors,
    normal_posteriors,
    dirichlet_posteriors,
    gamma_posteriors_all,
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


def estimate_probabilities(
    data: Union[List[List[Number]], np.ndarray], min_is_best: bool = False
) -> List[float]:
    """
    Estimate probabilities of being best for variants
    considering simulated data from respective posteriors.

    Parameters
    ----------
    data : List of simulated data for each variant.
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    if min_is_best:
        best_values = np.argmin(data, axis=0)
    else:
        best_values = np.argmax(data, axis=0)
    unique, counts = np.unique(best_values, return_counts=True)
    occurrences = dict(zip(unique, counts))
    sim_count = len(data[0])
    res = []
    for i in range(len(data)):
        res.append(round(occurrences.get(i, 0) / sim_count, 7))
    return res


def estimate_expected_loss(
    data: Union[List[List[Number]], np.ndarray], min_is_best: bool = False
) -> List[float]:
    """
    Estimate expected losses for variants considering simulated data from respective posteriors.

    Parameters
    ----------
    data : List of simulated data for each variant.
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res : List of expected loss for each variant.
    """
    if min_is_best:
        best_values = np.min(data, axis=0)
    else:
        best_values = np.max(data, axis=0)
    res = list(abs(np.mean(best_values - data, axis=1)).round(7))
    return res


def eval_bernoulli_agg(
    totals: List[int],
    positives: List[int],
    a_priors_beta: List[Number] = None,
    b_priors_beta: List[Number] = None,
    sim_count: int = 20000,
    seed: int = None,
    min_is_best: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for Beta-Bernoulli
    aggregated data per variant.

    Parameters
    ----------
    totals : List of total experiment observations (e.g. number of sessions) for each variant.
    positives : List of total number of ones (e.g. number of conversions) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_beta : List of prior alpha parameters of Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters of Beta distributions for each variant.
    seed : Random seed.
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    validate_bernoulli_input(totals, positives)

    if len(totals) == 0:
        return [], []

    # Default prior for all variants is Beta(0.5, 0.5) which is non-information prior.
    if not a_priors_beta:
        a_priors_beta = [0.5] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [0.5] * len(totals)

    beta_samples = beta_posteriors_all(
        totals, positives, sim_count, a_priors_beta, b_priors_beta, seed
    )

    res_pbbs = estimate_probabilities(beta_samples, min_is_best)
    res_loss = estimate_expected_loss(beta_samples, min_is_best)

    return res_pbbs, res_loss


def eval_normal_agg(
    totals: List[int],
    sums: List[float],
    sums_2: List[float],
    sim_count: int = 20000,
    m_priors: List[Number] = None,
    a_priors_ig: List[Number] = None,
    b_priors_ig: List[Number] = None,
    w_priors: List[Number] = None,
    seed: int = None,
    min_is_best: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for Normal
    aggregated data per variant.

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
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(totals) == 0:
        return [], []
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

    res_pbbs = estimate_probabilities(normal_samples, min_is_best)
    res_loss = estimate_expected_loss(normal_samples, min_is_best)

    return res_pbbs, res_loss


def eval_delta_lognormal_agg(
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
    min_is_best: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for Delta-Lognormal
    aggregated data per variant. For that reason, the method works with both totals and non_zeros.

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
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(totals) == 0:
        return [], []
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
        res_pbbs = list(np.full(len(totals), round(1 / len(totals), 7)))
        res_loss = [np.nan] * len(totals)
        return res_pbbs, res_loss
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

        res_pbbs = estimate_probabilities(combined_samples, min_is_best)
        res_loss = estimate_expected_loss(combined_samples, min_is_best)

        return res_pbbs, res_loss


def eval_numerical_dirichlet_agg(
    states: List[Union[float, int]],
    concentrations: List[List[int]],
    prior_alphas: List[List[Union[float, int]]] = None,
    sim_count: int = 20000,
    seed: int = None,
    min_is_best: bool = False,
):
    """
    Method estimating probabilities of being best and expected loss for Dirichlet-multinomial
    aggregated data per variant. States in this case are expected to be a numerical values
    (e.g. dice numbers, number of stars in a rating, etc.).

    Parameters
    ----------
    states : All possible outcomes in given multinomial distribution.
    concentrations : Concentration of observations for each state for all variants.
    prior_alphas : Prior alpha values for each state for all variants.
    sim_count : Number of simulations.
    seed : Random seed.
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(concentrations) == 0:
        return [], []

    # default prior will be expecting 1 observation in all states for all variants
    if not prior_alphas:
        prior_alphas = [[1] * len(states) for i in range(len(concentrations))]

    # we will need different generators for each call of dirichlet_posteriors
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(concentrations))

    means_samples = []
    for i in range(len(concentrations)):
        dir_post = dirichlet_posteriors(
            concentrations[i], prior_alphas[i], sim_count, child_seeds[i]
        )
        means = np.sum(np.multiply(dir_post, np.array(states)), axis=1)
        means_samples.append(list(means))

    res_pbbs = estimate_probabilities(means_samples, min_is_best)
    res_loss = estimate_expected_loss(means_samples, min_is_best)

    return res_pbbs, res_loss


def eval_poisson_agg(
    totals: List[int],
    sums: List[Union[float, int]],
    a_priors_gamma: List[Number] = None,
    b_priors_gamma: List[Number] = None,
    sim_count: int = 20000,
    seed: int = None,
    min_is_best: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for
    Poisson aggregated data per variant.

    Parameters
    ----------
    totals : List of total experiment observations (e.g. number of matches) for each variant.
    sums : List of sums of observations (e.g. number of goals) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_gamma : List of prior alpha parameters of Gamma distributions for each variant.
    b_priors_gamma : List of prior beta parameters of Gamma distributions for each variant.
    seed : Random seed.
    min_is_best : Option to change "being best" to a minimum. Default is maximum.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """

    if len(totals) == 0:
        return [], []

    # Default prior for all variants is Gamma(0.1, 0.1) which is on purpose quite vague.
    if not a_priors_gamma:
        a_priors_gamma = [0.1] * len(totals)
    if not b_priors_gamma:
        b_priors_gamma = [0.1] * len(totals)

    gamma_samples = gamma_posteriors_all(
        totals, sums, sim_count, a_priors_gamma, b_priors_gamma, seed
    )

    res_pbbs = estimate_probabilities(gamma_samples, min_is_best)
    res_loss = estimate_expected_loss(gamma_samples, min_is_best)

    return res_pbbs, res_loss


def eval_delta_normal_agg(
    totals: List[int],
    non_zeros: List[int],
    sums: List[float],
    sums_2: List[float],
    sim_count: int = 20000,
    a_priors_beta: List[Number] = None,
    b_priors_beta: List[Number] = None,
    m_priors: List[Number] = None,
    a_priors_ig: List[Number] = None,
    b_priors_ig: List[Number] = None,
    w_priors: List[Number] = None,
    seed: int = None,
    min_is_best: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for Delta-Normal
    aggregated data per variant. For that reason, the method works with both totals and non_zeros.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    non_zeros : List of numbers of non-zeros (e.g. number of conversions) for each variant.
    sums : List of sum of original data for each variant.
    sums_2 : List of sum of squared of original data for each variant.
    sim_count : Number of simulations.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    m_priors : List of prior means for each variant.
    a_priors_ig : List of prior alphas from inverse gamma dist approximating variance.
    b_priors_ig : List of prior betas from inverse gamma dist approximating variance.
    w_priors : List of prior effective sample sizes for each variant.
    seed : Random seed.
    min_is_best : Option to change "being best" to a minimum. Default is maximum.
    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(totals) == 0:
        return [], []
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
        res_pbbs = list(np.full(len(totals), round(1 / len(totals), 7)))
        res_loss = [np.nan] * len(totals)
        return res_pbbs, res_loss
    else:
        # we will need different generators for each call of normal_posteriors
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(len(totals) + 1)

        beta_samples = beta_posteriors_all(
            totals, non_zeros, sim_count, a_priors_beta, b_priors_beta, child_seeds[0]
        )

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
                    child_seeds[i + 1],
                )[0]
                for i in range(len(totals))
            ]
        )

        combined_samples = beta_samples * normal_samples

        res_pbbs = estimate_probabilities(combined_samples, min_is_best)
        res_loss = estimate_expected_loss(combined_samples, min_is_best)

        return res_pbbs, res_loss
