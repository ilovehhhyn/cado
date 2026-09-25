"""Group one source's trajectory records into training-step observations, with prompt groups as clusters.

Rollouts of one prompt share its difficulty, so they are not independent: evidence is counted per step, and a
step's clean-reward mean gets the cluster-robust (delta method) variance over its prompt groups,
    var(mean) = G / (G - 1) * sum_g (R_g - mean * n_g)^2 / (sum_g n_g)^2,
with R_g and n_g the clean-reward sum and count of group g. The step is the integer tag 'step' and the group the
tag 'group'; a record without a group tag is its own cluster. Unit batching instead cuts consecutive windows of
batch_units records, for runs that have no steps.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

from tplane.integrity.metrics import IntegrityError, is_non_agent_failure
from tplane.schema import Action, Record

STEP_TAG: Final[str] = "step"
GROUP_TAG: Final[str] = "group"
MIN_CLUSTERS: Final[int] = 2


class Batching(StrEnum):
    """How records become observations: one per training step, or one per window of units."""

    STEP = "step"
    UNITS = "units"


@dataclass(frozen=True, kw_only=True)
class StepObservation:
    """One step of one source: every attempt, its infra or grader failures, and the clean-reward mean."""

    label: int
    last_unit_id: str
    at_ms: int
    units: int
    failures: int
    clusters: int
    clean_count: int
    clean_mean: float | None
    clean_variance_of_mean: (
        float | None
    )  # None with fewer than 2 clusters: the variance is not identifiable
    clean_unit_sd: float | None
    failure_variance_of_mean: (
        float | None
    )  # cluster-robust variance of the failure proportion; None below 2 clusters


def step_observations(
    units: Sequence[Record], *, batching: Batching, batch_units: int
) -> list[StepObservation]:
    """Return one observation per step (or per window of batch_units), in step order; units must be in start order."""
    match batching:
        case Batching.STEP:
            steps: dict[int, list[Record]] = {}
            for record in units:
                steps.setdefault(_step_of(record), []).append(record)
            return [_observe(label, steps[label]) for label in sorted(steps)]
        case Batching.UNITS:
            return [
                _observe(start // batch_units, list(units[start : start + batch_units]))
                for start in range(0, len(units), batch_units)
            ]
        case _:
            assert_never(batching)


def clean_reward(record: Record) -> float | None:
    """Return the reward of a final attempt that did not fail for an infra or grader cause."""
    if record.decision.action is Action.RETRY or is_non_agent_failure(record):
        return None
    return record.reward


def _observe(label: int, members: list[Record]) -> StepObservation:
    last = max(members, key=lambda record: (record.started_at_ms, record.unit_id))
    groups: dict[str, list[float]] = {}
    failures: dict[str, list[float]] = {}
    for record in members:
        failures.setdefault(_group_of(record), []).append(
            1.0 if is_non_agent_failure(record) else 0.0
        )
        reward = clean_reward(record)
        if reward is not None:
            groups.setdefault(_group_of(record), []).append(reward)
    failure_count = sum(sum(values) for values in failures.values())
    rewards = [reward for values in groups.values() for reward in values]
    mean = statistics.fmean(rewards) if len(rewards) != 0 else None
    return StepObservation(
        label=label,
        last_unit_id=last.unit_id,
        at_ms=last.started_at_ms,
        units=len(members),
        failures=int(failure_count),
        clusters=len({_group_of(record) for record in members}),
        clean_count=len(rewards),
        clean_mean=mean,
        clean_variance_of_mean=None
        if mean is None
        else _cluster_variance(list(groups.values()), mean),
        clean_unit_sd=statistics.stdev(rewards) if len(rewards) >= 2 else None,
        failure_variance_of_mean=_cluster_variance(
            list(failures.values()), failure_count / len(members)
        ),
    )


def _cluster_variance(groups: list[list[float]], mean: float) -> float | None:
    if len(groups) < MIN_CLUSTERS:
        return None
    total = sum(len(values) for values in groups)
    squared = math.fsum((math.fsum(values) - mean * len(values)) ** 2 for values in groups)
    return len(groups) / (len(groups) - 1) * squared / total**2


def _step_of(record: Record) -> int:
    value = dict(record.tags).get(STEP_TAG)
    if value is None or not value.lstrip("-").isdigit():
        raise IntegrityError(
            f"{record.unit_id} has no integer '{STEP_TAG}' tag; tag each trajectory with its training step, "
            "or use batching=units"
        )
    return int(value)


def _group_of(record: Record) -> str:
    return dict(record.tags).get(GROUP_TAG, "unit:" + record.unit_id)
