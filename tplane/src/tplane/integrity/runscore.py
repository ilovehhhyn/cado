"""Score whole runs against a profile of normal runs, for faults that are present from a run's first step.

For feature f at step t, z_t = (x_t - m[f, t]) / s[f]: m is the per-step mean over normal runs and s one pooled spread
per feature, s[f]^2 = mean over steps of the per-step variance, floored at 1% of the mean level. One spread per
feature is far more stable than one per step when few normal runs exist; on RFT-FaultBench, per-step spreads from
about nine runs were noise (docs/results/rft-faultbench.md). A shift delta ~ N(0, tau^2) held over the n observed
steps has the Bayes factor
    log BF_f = -1/2 log(1 + n tau^2) + (sum_t z_t)^2 tau^2 / (2 (1 + n tau^2)),
and the run score is log mean_f BF_f: one unknown feature is shifted. The alarm threshold is the mean plus k
standard deviations of normal scores computed out of sample, each fold scored by a profile built without it.
Grouping runs by configuration (task, batch size) is the caller's job: build one profile per group.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

MIN_NORMAL_RUNS: Final[int] = 2
RELATIVE_SPREAD_FLOOR: Final[float] = (
    0.01  # a spread below 1% of the level is treated as 1% of the level
)
ABSOLUTE_SPREAD_FLOOR: Final[float] = 1e-9  # keeps z finite for a feature that is always exactly 0
DEFAULT_PRIOR_SCALE: Final[float] = (
    0.3  # prior sd of the shift, in spreads; chosen on RFT-FaultBench
)

RunSeries: TypeAlias = Mapping[str, Sequence[float | None]]


class RunScoreError(ValueError):
    """A run score needs a profile from at least two normal runs and a run that overlaps it."""


@dataclass(frozen=True, kw_only=True)
class RunProfile:
    """Per-step means and one pooled spread per feature, from normal runs."""

    means: Mapping[tuple[str, int], float]
    spreads: Mapping[str, float]
    normal_runs: int


def build_profile(normals: Sequence[RunSeries]) -> RunProfile:
    """Return the per-step mean of every feature and its pooled, floored spread."""
    if len(normals) < MIN_NORMAL_RUNS:
        raise RunScoreError(
            f"a profile needs at least {MIN_NORMAL_RUNS} normal runs, got {len(normals)}"
        )
    columns: dict[tuple[str, int], list[float]] = {}
    for run in normals:
        for feature, series in run.items():
            for step, value in enumerate(series):
                if value is not None:
                    columns.setdefault((feature, step), []).append(value)
    means = {key: statistics.fmean(values) for key, values in sorted(columns.items())}
    spreads = {
        feature: _pooled_spread(feature, columns, means)
        for feature in sorted({key[0] for key in columns})
    }
    return RunProfile(means=means, spreads=spreads, normal_runs=len(normals))


def score_run(
    run: RunSeries, profile: RunProfile, *, prior_scale: float = DEFAULT_PRIOR_SCALE
) -> float:
    """Return log mean_f BF_f over the features the run shares with the profile."""
    if not prior_scale > 0:
        raise RunScoreError(f"prior_scale must be positive, got {prior_scale}")
    evidence = []
    for feature in sorted(run):
        if feature not in profile.spreads:
            continue
        z = [
            (value - profile.means[(feature, step)]) / profile.spreads[feature]
            for step, value in enumerate(run[feature])
            if value is not None and (feature, step) in profile.means
        ]
        if len(z) != 0:
            evidence.append(_shift_log_bayes_factor(z, prior_scale))
    if len(evidence) == 0:
        raise RunScoreError(
            "run shares no feature and step with the profile; group runs by configuration first"
        )
    largest = max(evidence)
    return largest + math.log(
        math.fsum(math.exp(value - largest) for value in evidence) / len(evidence)
    )


def out_of_sample_scores(
    normals: Sequence[RunSeries], *, folds: int = 5, prior_scale: float = DEFAULT_PRIOR_SCALE
) -> tuple[float, ...]:
    """Score each normal run with a profile built without its fold (fold j holds runs j, j + folds, ...); folds = len(normals) is leave-one-out."""
    smallest_rest = len(normals) - math.ceil(len(normals) / folds)
    if folds < 2 or smallest_rest < MIN_NORMAL_RUNS:
        raise RunScoreError(
            f"{folds} folds over {len(normals)} normal runs leave {smallest_rest} run per profile; "
            f"at least {MIN_NORMAL_RUNS} are needed"
        )
    scores: list[float] = [0.0] * len(normals)
    for fold in range(folds):
        profile = build_profile([run for index, run in enumerate(normals) if index % folds != fold])
        for index in range(fold, len(normals), folds):
            scores[index] = score_run(normals[index], profile, prior_scale=prior_scale)
    return tuple(scores)


def calibrate_threshold(normal_scores: Sequence[float], *, multiplier: float) -> float:
    """Return mean + multiplier * sd of out-of-sample normal scores; a run above it is flagged."""
    if len(normal_scores) < MIN_NORMAL_RUNS:
        raise RunScoreError(
            f"calibration needs at least {MIN_NORMAL_RUNS} normal scores, got {len(normal_scores)}"
        )
    if multiplier < 0:
        raise RunScoreError(f"multiplier must be non-negative, got {multiplier}")
    return statistics.fmean(normal_scores) + multiplier * statistics.stdev(normal_scores)


def _pooled_spread(
    feature: str,
    columns: Mapping[tuple[str, int], list[float]],
    means: Mapping[tuple[str, int], float],
) -> float:
    steps = [key for key in columns if key[0] == feature]
    variances = [
        statistics.variance(columns[key]) for key in steps if len(columns[key]) >= MIN_NORMAL_RUNS
    ]
    pooled = math.sqrt(statistics.fmean(variances)) if len(variances) != 0 else 0.0
    level = statistics.fmean(abs(means[key]) for key in steps)
    return max(pooled, RELATIVE_SPREAD_FLOOR * level, ABSOLUTE_SPREAD_FLOOR)


def _shift_log_bayes_factor(z: Sequence[float], prior_scale: float) -> float:
    n, total, variance = len(z), math.fsum(z), prior_scale * prior_scale
    return -0.5 * math.log1p(n * variance) + total * total * variance / (2 * (1 + n * variance))
