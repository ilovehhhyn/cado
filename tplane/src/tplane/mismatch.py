"""Estimate train-inference mismatch from per-token log-probabilities, with sequences as the independent units.

For a token y_t drawn from the rollout engine q, with trainer probability p:
    delta_t = clamp(log p(y_t) - log q(y_t), -20, 20),  rho_t = exp(delta_t)
    E_q[rho_t] = 1                              (exact when p and q share support)
    E_q[rho_t - 1 - delta_t] = KL(q || p)        (the k3 estimator, non-negative per token)
Top-p or top-k truncation in q without renormalising p gives E_q[rho_t] = p(nucleus) < 1 and biases k3 low,
so the mean ratio detects a support mismatch that the KL alone misses.

Over n sequences with T_i tokens and sums Y_i, the global token mean r = sum(Y_i) / sum(T_i) has the
cluster-robust (delta method) variance  n / (n - 1) * sum((Y_i - r * T_i)^2) / sum(T_i)^2.
References: verl rollout_corr_helper.compute_offpolicy_metrics (global token mean of k3);
slime examples/train_infer_mismatch_helper/mis.py (SAFETY_BOUND = 20).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tplane.schema import MIN_MISMATCH_SEQUENCES, MismatchSummary

SAFETY_BOUND_LOG_RATIO: Final[float] = 20.0  # slime's SAFETY_BOUND; keeps exp(delta) finite


class MismatchError(ValueError):
    """Mismatch inputs are matched, finite log-probabilities over at least two non-empty sequences."""


@dataclass(frozen=True, kw_only=True)
class SequenceSums:
    """Per-sequence sums a trainer can compute in its own tensor library: tokens, sum of rho, sum of k3."""

    tokens: int
    ratio_sum: float
    k3_sum: float


def sequence_sums(
    *, train_logprobs: Sequence[float], rollout_logprobs: Sequence[float]
) -> SequenceSums:
    """Sum rho_t and k3_t = expm1(delta_t) - delta_t over the response tokens of one sequence."""
    if len(train_logprobs) != len(rollout_logprobs):
        raise MismatchError(
            f"train_logprobs has {len(train_logprobs)} tokens but rollout_logprobs has {len(rollout_logprobs)}"
        )
    if len(train_logprobs) == 0:
        raise MismatchError("a sequence must have at least one token")
    ratio_sum = 0.0
    k3_sum = 0.0
    for train, rollout in zip(train_logprobs, rollout_logprobs, strict=True):
        for value in (train, rollout):
            if not math.isfinite(value) or value > 0:
                raise MismatchError(
                    f"log-probabilities must be finite and at most 0, got {value!r}"
                )
        delta = min(max(train - rollout, -SAFETY_BOUND_LOG_RATIO), SAFETY_BOUND_LOG_RATIO)
        ratio_sum += math.exp(delta)
        # expm1 keeps precision for small delta; max() absorbs a rounding error below zero.
        k3_sum += max(0.0, math.expm1(delta) - delta)
    return SequenceSums(tokens=len(train_logprobs), ratio_sum=ratio_sum, k3_sum=k3_sum)


def summarise_mismatch(sequences: Sequence[SequenceSums]) -> MismatchSummary:
    """Reduce per-sequence sums to global token means of k3 and rho with sequence-clustered standard errors."""
    if len(sequences) < MIN_MISMATCH_SEQUENCES:
        raise MismatchError(
            f"at least {MIN_MISMATCH_SEQUENCES} sequences are required for a standard error, got {len(sequences)}"
        )
    for index, item in enumerate(sequences):
        _validate_sums(index, item)
    tokens = [item.tokens for item in sequences]
    kl, kl_error = _ratio_estimate([item.k3_sum for item in sequences], tokens)
    ratio, ratio_error = _ratio_estimate([item.ratio_sum for item in sequences], tokens)
    return MismatchSummary(
        sequences=len(sequences),
        tokens=sum(tokens),
        kl=kl,
        kl_standard_error=kl_error,
        ratio_mean=ratio,
        ratio_standard_error=ratio_error,
    )


def _validate_sums(index: int, item: SequenceSums) -> None:
    if item.tokens < 1:
        raise MismatchError(f"sequence {index}: tokens must be at least 1, got {item.tokens}")
    sums = (item.ratio_sum, item.k3_sum)
    if any(not math.isfinite(value) or value < 0 for value in sums):
        raise MismatchError(
            f"sequence {index}: ratio_sum and k3_sum must be finite and non-negative, got {sums!r}"
        )


def _ratio_estimate(totals: Sequence[float], tokens: Sequence[int]) -> tuple[float, float]:
    """Return r = sum(Y) / sum(T) and its cluster-robust standard error."""
    count = len(totals)
    token_total = sum(tokens)
    estimate = math.fsum(totals) / token_total
    squared = math.fsum(
        (total - estimate * size) ** 2 for total, size in zip(totals, tokens, strict=True)
    )
    variance = count / (count - 1) * squared / token_total**2
    return estimate, math.sqrt(variance)
