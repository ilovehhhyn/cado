"""Mismatch estimators satisfy the importance-sampling identities: E_q[p/q] = 1 and E_q[p/q - 1 - log(p/q)] = KL(q||p), with sequence-clustered standard errors.

Oracle: the exact KL and nucleus mass of small categorical distributions, computed directly; tokens are drawn with a seeded random.Random.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from tplane.mismatch import (
    SAFETY_BOUND_LOG_RATIO,
    MismatchError,
    SequenceSums,
    sequence_sums,
    summarise_mismatch,
)
from tplane.schema import MIN_MISMATCH_SEQUENCES

VOCABULARY = 8


def categorical(rng: random.Random) -> list[float]:
    weights = [rng.random() + 0.05 for _ in range(VOCABULARY)]
    total = sum(weights)
    return [weight / total for weight in weights]


def exact_kl(q: list[float], p: list[float]) -> float:
    return sum(qi * math.log(qi / pi) for qi, pi in zip(q, p, strict=True) if qi > 0)


def simulate(
    rng: random.Random, *, sampler: list[float], trainer: list[float], sequences: int, length: int
) -> list[SequenceSums]:
    indices = list(range(VOCABULARY))
    sums: list[SequenceSums] = []
    for _ in range(sequences):
        tokens = rng.choices(indices, weights=sampler, k=length)
        sums.append(
            sequence_sums(
                train_logprobs=[math.log(trainer[token]) for token in tokens],
                rollout_logprobs=[math.log(sampler[token]) for token in tokens],
            )
        )
    return sums


def test_sequence_sums_compute_ratio_and_k3_per_token() -> None:
    sums = sequence_sums(
        train_logprobs=[math.log(0.5), math.log(0.2)],
        rollout_logprobs=[math.log(0.25), math.log(0.4)],
    )

    assert sums.tokens == 2
    assert sums.ratio_sum == pytest.approx(2.0 + 0.5, rel=1e-12, abs=0.0)
    assert sums.k3_sum == pytest.approx(
        (2.0 - 1 - math.log(2.0)) + (0.5 - 1 - math.log(0.5)), rel=1e-12, abs=0.0
    )


def test_identical_logprobs_give_zero_kl_and_unit_ratio_exactly() -> None:
    logprobs = [-0.1, -2.0, -0.7]
    sums = [sequence_sums(train_logprobs=logprobs, rollout_logprobs=logprobs) for _ in range(3)]

    summary = summarise_mismatch(sums)

    assert (summary.kl, summary.kl_standard_error) == (0.0, 0.0)
    assert (summary.ratio_mean, summary.ratio_standard_error) == (1.0, 0.0)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_mean_ratio_is_one_and_kl_matches_the_exact_divergence_under_full_support(
    seed: int,
) -> None:
    rng = random.Random(seed)
    sampler, trainer = categorical(rng), categorical(rng)

    summary = summarise_mismatch(
        simulate(rng, sampler=sampler, trainer=trainer, sequences=2_000, length=20)
    )

    assert abs(summary.ratio_mean - 1.0) < 4 * summary.ratio_standard_error, summary
    assert abs(summary.kl - exact_kl(sampler, trainer)) < 4 * summary.kl_standard_error, summary


def test_top_p_truncation_without_renormalising_the_trainer_lowers_the_mean_ratio_to_the_nucleus_mass() -> (
    None
):
    # Invariant: E_q[p/q] = p(N) < 1 when the sampler draws only from a nucleus N of the trainer's distribution.
    rng = random.Random(7)
    trainer = categorical(rng)
    engine = [
        0.8 * probability + 0.2 * noise
        for probability, noise in zip(trainer, categorical(rng), strict=True)
    ]
    nucleus = sorted(range(VOCABULARY), key=lambda token: -engine[token])[:5]
    engine_mass = sum(engine[token] for token in nucleus)
    sampler = [
        engine[token] / engine_mass if token in nucleus else 0.0 for token in range(VOCABULARY)
    ]
    nucleus_mass = sum(trainer[token] for token in nucleus)

    summary = summarise_mismatch(
        simulate(rng, sampler=sampler, trainer=trainer, sequences=2_000, length=20)
    )

    assert nucleus_mass < 0.9
    assert abs(summary.ratio_mean - nucleus_mass) < 4 * summary.ratio_standard_error, summary
    assert (1.0 - summary.ratio_mean) > 10 * summary.ratio_standard_error, summary


def test_equal_length_sequences_give_the_standard_error_of_per_sequence_means() -> None:
    # Oracle: with equal lengths the cluster ratio estimator is the mean of sequence means.
    rng = random.Random(3)
    sampler, trainer = categorical(rng), categorical(rng)
    sums = simulate(rng, sampler=sampler, trainer=trainer, sequences=50, length=10)
    sequence_means = [item.k3_sum / item.tokens for item in sums]

    summary = summarise_mismatch(sums)

    assert summary.kl == pytest.approx(statistics.fmean(sequence_means), rel=1e-12, abs=1e-15)
    expected_error = statistics.stdev(sequence_means) / math.sqrt(len(sequence_means))
    assert summary.kl_standard_error == pytest.approx(expected_error, rel=1e-9, abs=1e-15)


def test_log_ratio_is_clamped_to_the_safety_bound() -> None:
    sums = sequence_sums(train_logprobs=[0.0], rollout_logprobs=[-1_000.0])

    assert sums.ratio_sum == pytest.approx(math.exp(SAFETY_BOUND_LOG_RATIO), rel=1e-12, abs=0.0)


def test_sequence_sums_reject_mismatched_or_empty_or_positive_logprobs() -> None:
    with pytest.raises(
        MismatchError, match="train_logprobs has 2 tokens but rollout_logprobs has 1"
    ):
        sequence_sums(train_logprobs=[-1.0, -1.0], rollout_logprobs=[-1.0])
    with pytest.raises(MismatchError, match="a sequence must have at least one token"):
        sequence_sums(train_logprobs=[], rollout_logprobs=[])
    with pytest.raises(
        MismatchError, match="log-probabilities must be finite and at most 0, got 0.5"
    ):
        sequence_sums(train_logprobs=[0.5], rollout_logprobs=[-1.0])


def test_summary_rejects_too_few_sequences() -> None:
    one = sequence_sums(train_logprobs=[-1.0], rollout_logprobs=[-1.0])

    with pytest.raises(
        MismatchError,
        match=f"at least {MIN_MISMATCH_SEQUENCES} sequences are required for a standard error, got 1",
    ):
        summarise_mismatch([one])


def test_summary_rejects_invalid_sums_from_a_trainer() -> None:
    good = SequenceSums(tokens=2, ratio_sum=2.0, k3_sum=0.0)

    with pytest.raises(MismatchError, match="sequence 1: tokens must be at least 1, got 0"):
        summarise_mismatch([good, SequenceSums(tokens=0, ratio_sum=0.0, k3_sum=0.0)])
    with pytest.raises(
        MismatchError, match="sequence 1: ratio_sum and k3_sum must be finite and non-negative"
    ):
        summarise_mismatch([good, SequenceSums(tokens=1, ratio_sum=float("nan"), k3_sum=0.0)])
