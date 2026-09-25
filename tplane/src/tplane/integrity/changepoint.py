"""Track the Bayesian posterior probability that a change has happened, and the log-likelihood ratios that feed it.

Shiryaev procedure (Veeravalli and Banerjee, "Quickest Change Detection", arXiv 1210.5552, eq. 13-14).
With a geometric prior P(change at n) = rho * (1 - rho)^(n - 1) and L_n = f_changed(x_n) / f_healthy(x_n):
    p~ = p_{n-1} + (1 - p_{n-1}) * rho
    p_n = p~ * L_n / (p~ * L_n + 1 - p~)
Stopping when p_n >= 1 - alpha gives P(alarm before the change) <= alpha (Theorem 3.2), provided rho and both
likelihoods are right. The state is the log-odds log(p / (1 - p)), so evidence of any size never overflows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

NO_CHANGE_LOG_ODDS: Final[float] = -math.inf


class ChangepointError(ValueError):
    """Detector inputs are finite, with probabilities and rates strictly inside (0, 1)."""


def shiryaev_update(log_odds: float, *, log_likelihood_ratio: float, hazard: float) -> float:
    """Return logit(p_n) from logit(p_{n-1}): logit(p~) + log L_n, with log(1 - p~) = log(1 - p) + log(1 - rho)."""
    if not 0 < hazard < 1:
        raise ChangepointError(f"hazard must be in (0, 1), got {hazard}")
    if not math.isfinite(log_likelihood_ratio):
        raise ChangepointError(f"log_likelihood_ratio must be finite, got {log_likelihood_ratio}")
    log_not_changed = -_softplus(log_odds) + math.log1p(-hazard)
    # log(1 - exp(x)) as log(-expm1(x)) stays exact when x is near 0, as it is for a tiny hazard.
    log_changed = math.log(-math.expm1(log_not_changed))
    return log_changed - log_not_changed + log_likelihood_ratio


def probability_of(log_odds: float) -> float:
    """Return p = 1 / (1 + exp(-log_odds)) without overflow."""
    if log_odds >= 0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    odds = math.exp(log_odds)
    return odds / (1.0 + odds)


def log_odds_of(probability: float) -> float:
    """Return log(p / (1 - p)) for a probability strictly inside (0, 1)."""
    if not 0 < probability < 1:
        raise ChangepointError(f"probability must be in (0, 1), got {probability}")
    return math.log(probability) - math.log1p(-probability)


def bernoulli_llr(event: bool, *, rate_healthy: float, rate_changed: float) -> float:
    """Return log(p1 / p0) for an event and log((1 - p1) / (1 - p0)) otherwise."""
    if not 0 < rate_healthy < rate_changed < 1:
        raise ChangepointError(
            f"rates must satisfy 0 < rate_healthy < rate_changed < 1, got {rate_healthy} and {rate_changed}"
        )
    if event:
        return math.log(rate_changed) - math.log(rate_healthy)
    return math.log1p(-rate_changed) - math.log1p(-rate_healthy)


def gaussian_llr(
    x: float, *, mean_healthy: float, mean_changed: float, standard_deviation: float
) -> float:
    """Return log N(x; m1, s) - log N(x; m0, s) = (m1 - m0) / s^2 * (x - (m0 + m1) / 2)."""
    if not standard_deviation > 0:
        raise ChangepointError(f"standard_deviation must be positive, got {standard_deviation}")
    # Dividing by s twice, not by s^2, keeps a tiny s from underflowing to zero.
    shift = (mean_changed - mean_healthy) / standard_deviation
    return shift * ((x - (mean_healthy + mean_changed) / 2) / standard_deviation)


def student_t_llr(
    x: float, *, mean_healthy: float, mean_changed: float, scale: float, degrees: float
) -> float:
    """Return log t_nu(x; m1, s) - log t_nu(x; m0, s) = -(nu + 1)/2 [log(1 + z1^2/nu) - log(1 + z0^2/nu)].

    The t predictive is the exact Bayesian predictive of a normal observation whose mean and variance were
    estimated from nu + 1 earlier ones. Its ratio is bounded, so one extreme observation cannot decide a change.
    """
    if not (scale > 0 and degrees >= 1):
        raise ChangepointError(
            f"scale must be positive and degrees at least 1, got {scale} and {degrees}"
        )
    changed = (x - mean_changed) / scale
    healthy = (x - mean_healthy) / scale
    return (
        -(degrees + 1)
        / 2
        * (math.log1p(changed * changed / degrees) - math.log1p(healthy * healthy / degrees))
    )


def log_mean_exp(values: Sequence[float]) -> float:
    """Return log(mean(exp(v))), the log-likelihood ratio of an equal mixture of alternatives with these ratios."""
    if len(values) == 0:
        raise ChangepointError("log_mean_exp needs at least one value")
    largest = max(values)
    return largest + math.log(
        math.fsum(math.exp(value - largest) for value in values) / len(values)
    )


def _softplus(x: float) -> float:
    """Return log(1 + exp(x)), exact at -inf and +inf."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))
