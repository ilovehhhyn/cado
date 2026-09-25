"""The Shiryaev posterior equals the exact Bayes posterior of a change under a geometric prior, and alarming at 1 - alpha keeps false alarms at or below alpha.

Oracle: brute-force enumeration of every change time with its prior weight and likelihood.
"""

from __future__ import annotations

import math
import random

import pytest

from tplane.integrity.changepoint import (
    NO_CHANGE_LOG_ODDS,
    ChangepointError,
    bernoulli_llr,
    gaussian_llr,
    log_mean_exp,
    log_odds_of,
    probability_of,
    shiryaev_update,
    student_t_llr,
)


def brute_force_posterior(llrs: list[float], hazard: float) -> float:
    """P(change at or before n | data) with P(change at k) = hazard * (1 - hazard)^(k - 1)."""
    n = len(llrs)
    weights: list[float] = []
    for change_at in range(1, n + 2):  # n + 1 stands for "after the data"
        prior = hazard * (1 - hazard) ** (change_at - 1) if change_at <= n else (1 - hazard) ** n
        weights.append(prior * math.exp(sum(llrs[change_at - 1 :])) if change_at <= n else prior)
    return sum(weights[:n]) / sum(weights)


@pytest.mark.parametrize("seed", (0, 1, 2, 3))
def test_recursion_matches_brute_force_bayes_on_bernoulli_data(seed: int) -> None:
    rng = random.Random(seed)
    events = [rng.random() < (0.05 if index < 15 else 0.4) for index in range(30)]
    llrs = [bernoulli_llr(event, rate_healthy=0.05, rate_changed=0.4) for event in events]
    hazard = 0.02

    log_odds = NO_CHANGE_LOG_ODDS
    for count, llr in enumerate(llrs, start=1):
        log_odds = shiryaev_update(log_odds, log_likelihood_ratio=llr, hazard=hazard)

        assert probability_of(log_odds) == pytest.approx(
            brute_force_posterior(llrs[:count], hazard), rel=1e-9, abs=1e-12
        )


def test_no_evidence_raises_the_posterior_by_exactly_the_prior_hazard() -> None:
    log_odds = shiryaev_update(NO_CHANGE_LOG_ODDS, log_likelihood_ratio=0.0, hazard=0.01)

    assert probability_of(log_odds) == pytest.approx(0.01, rel=1e-12, abs=0.0)


def test_extreme_evidence_saturates_without_overflow() -> None:
    high = shiryaev_update(0.0, log_likelihood_ratio=5_000.0, hazard=1e-3)
    low = shiryaev_update(0.0, log_likelihood_ratio=-5_000.0, hazard=1e-3)
    again = shiryaev_update(high, log_likelihood_ratio=-1.0, hazard=1e-3)

    assert probability_of(high) == 1.0 and math.isfinite(high)
    assert probability_of(low) < 1e-3 and math.isfinite(low)
    assert math.isfinite(again)


def test_alarming_at_one_minus_alpha_keeps_false_alarms_below_alpha() -> None:
    # Invariant: Shiryaev's bound P(alarm before the change) <= alpha when the prior and likelihoods are true.
    alpha, hazard, trials = 0.05, 0.05, 2_000
    rng = random.Random(11)
    threshold = log_odds_of(1 - alpha)
    false_alarms = 0
    for _ in range(trials):
        change_at = 1
        while rng.random() >= hazard:
            change_at += 1
        log_odds = NO_CHANGE_LOG_ODDS
        for index in range(1, 10_000):
            event = rng.random() < (0.3 if index >= change_at else 0.1)
            llr = bernoulli_llr(event, rate_healthy=0.1, rate_changed=0.3)
            log_odds = shiryaev_update(log_odds, log_likelihood_ratio=llr, hazard=hazard)
            if log_odds >= threshold:
                false_alarms += index < change_at
                break

    rate = false_alarms / trials
    assert rate <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / trials), rate


def test_gaussian_llr_equals_the_difference_of_log_densities() -> None:
    def log_density(x: float, mean: float, sd: float) -> float:
        return -0.5 * math.log(2 * math.pi * sd * sd) - (x - mean) ** 2 / (2 * sd * sd)

    for x in (-1.0, 0.0, 0.3, 2.5):
        expected = log_density(x, 1.0, 0.7) - log_density(x, 0.2, 0.7)

        assert gaussian_llr(
            x, mean_healthy=0.2, mean_changed=1.0, standard_deviation=0.7
        ) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_log_mean_exp_is_the_log_of_the_mean_ratio_and_does_not_overflow() -> None:
    assert log_mean_exp([math.log(2.0), math.log(6.0)]) == pytest.approx(
        math.log(4.0), rel=1e-12, abs=0.0
    )
    assert log_mean_exp([1_000.0, 1_000.0]) == pytest.approx(1_000.0, rel=1e-12, abs=0.0)
    with pytest.raises(ChangepointError, match="log_mean_exp needs at least one value"):
        log_mean_exp([])


def test_bernoulli_llr_matches_the_log_rate_ratio() -> None:
    assert bernoulli_llr(True, rate_healthy=0.02, rate_changed=0.16) == pytest.approx(
        math.log(8), rel=1e-12, abs=0.0
    )
    assert bernoulli_llr(False, rate_healthy=0.02, rate_changed=0.16) == pytest.approx(
        math.log(0.84 / 0.98), rel=1e-12, abs=0.0
    )


def test_invalid_parameters_are_rejected_with_the_rule() -> None:
    with pytest.raises(ChangepointError, match="hazard must be in \\(0, 1\\), got 0"):
        shiryaev_update(0.0, log_likelihood_ratio=0.0, hazard=0)
    with pytest.raises(ChangepointError, match="log_likelihood_ratio must be finite, got nan"):
        shiryaev_update(0.0, log_likelihood_ratio=float("nan"), hazard=0.1)
    with pytest.raises(
        ChangepointError,
        match="rates must satisfy 0 < rate_healthy < rate_changed < 1, got 0.2 and 0.1",
    ):
        bernoulli_llr(True, rate_healthy=0.2, rate_changed=0.1)
    with pytest.raises(ChangepointError, match="standard_deviation must be positive, got 0"):
        gaussian_llr(0.0, mean_healthy=0.0, mean_changed=1.0, standard_deviation=0)
    with pytest.raises(ChangepointError, match="probability must be in \\(0, 1\\), got 1"):
        log_odds_of(1)


def test_a_tiny_hazard_does_not_underflow_to_a_domain_error() -> None:
    log_odds = shiryaev_update(NO_CHANGE_LOG_ODDS, log_likelihood_ratio=0.0, hazard=1e-18)

    assert probability_of(log_odds) == pytest.approx(1e-18, rel=1e-9, abs=0.0)


def log_t_density(x: float, location: float, scale: float, degrees: float) -> float:
    """Oracle: the Student-t log density written out with lgamma, independent of student_t_llr."""
    z = (x - location) / scale
    return (
        math.lgamma((degrees + 1) / 2)
        - math.lgamma(degrees / 2)
        - 0.5 * math.log(degrees * math.pi)
        - math.log(scale)
        - (degrees + 1) / 2 * math.log1p(z * z / degrees)
    )


def test_student_t_llr_equals_the_difference_of_log_densities() -> None:
    for x in (-3.0, 0.1, 0.4, 0.5, 2.0):
        expected = log_t_density(x, 0.375, 0.125, 19) - log_t_density(x, 0.5, 0.125, 19)

        assert student_t_llr(
            x, mean_healthy=0.5, mean_changed=0.375, scale=0.125, degrees=19
        ) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_student_t_llr_approaches_the_gaussian_llr_as_degrees_grow() -> None:
    gaussian = gaussian_llr(0.4, mean_healthy=0.5, mean_changed=0.375, standard_deviation=0.125)

    t = student_t_llr(0.4, mean_healthy=0.5, mean_changed=0.375, scale=0.125, degrees=1e7)

    assert t == pytest.approx(gaussian, rel=1e-5, abs=1e-9)


def test_student_t_evidence_from_one_extreme_step_is_bounded_but_a_real_drop_still_counts() -> None:
    moderate = student_t_llr(0.375, mean_healthy=0.5, mean_changed=0.375, scale=0.125, degrees=19)
    gross = student_t_llr(
        0.5 - 4 * 0.125, mean_healthy=0.5, mean_changed=0.375, scale=0.125, degrees=19
    )
    absurd = student_t_llr(-100.0, mean_healthy=0.5, mean_changed=0.375, scale=0.125, degrees=19)

    assert 0.4 < moderate < 0.6
    assert gross > 2.0
    assert 0 < absurd < 1.0


def test_student_t_llr_rejects_invalid_scale_or_degrees() -> None:
    with pytest.raises(
        ChangepointError, match="scale must be positive and degrees at least 1, got 0 and 19"
    ):
        student_t_llr(0.0, mean_healthy=0.0, mean_changed=1.0, scale=0, degrees=19)
