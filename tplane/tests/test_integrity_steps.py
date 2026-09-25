"""Trajectory records become one observation per training step, with prompt groups as clusters for the variance of the step mean and the design effect of the failure rate.

Oracle: with equal-size groups the cluster-robust variance of the mean equals the variance of the group means over G.
"""

from __future__ import annotations

import statistics

import pytest

from tests.support import make_unit_record
from tplane.integrity.metrics import IntegrityError
from tplane.integrity.monitor import design_effect
from tplane.integrity.steps import Batching, step_observations
from tplane.schema import Action, FailureKind, Record


def step(
    rewards_by_group: list[list[float]], *, step_index: int = 0, start: int = 0
) -> list[Record]:
    records, index = [], start
    for group, rewards in enumerate(rewards_by_group):
        for reward in rewards:
            records.append(
                make_unit_record(index, reward=reward, step=step_index, group=f"g{group}")
            )
            index += 1
    return records


def test_the_step_mean_variance_is_the_cluster_robust_variance_over_prompt_groups() -> None:
    groups = [[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]

    (observation,) = step_observations(step(groups), batching=Batching.STEP, batch_units=32)

    group_means = [statistics.fmean(values) for values in groups]
    assert observation.clean_mean == pytest.approx(
        statistics.fmean(group_means), rel=1e-12, abs=0.0
    )
    assert observation.clean_variance_of_mean == pytest.approx(
        statistics.variance(group_means) / 3, rel=1e-12, abs=0.0
    )
    assert (observation.units, observation.clusters, observation.clean_count) == (12, 3, 12)


def test_steps_are_ordered_by_their_tag_and_windows_by_start_order() -> None:
    records = step([[1.0, 0.0]], step_index=7, start=0) + step([[0.0, 0.0]], step_index=3, start=2)

    by_step = step_observations(
        sorted(records, key=lambda record: record.started_at_ms),
        batching=Batching.STEP,
        batch_units=32,
    )
    by_units = step_observations(records, batching=Batching.UNITS, batch_units=3)

    assert [observation.label for observation in by_step] == [3, 7]
    assert [observation.units for observation in by_units] == [3, 1]


def test_infra_and_grader_failures_are_counted_but_never_enter_the_clean_mean() -> None:
    records = [
        make_unit_record(0, reward=1.0, step=0, group="a"),
        make_unit_record(
            1, failure_kind=FailureKind.SANDBOX, action=Action.ZERO, step=0, group="a"
        ),
        make_unit_record(
            2, failure_kind=FailureKind.AGENT_EXCEPTION, action=Action.ZERO, step=0, group="b"
        ),
    ]

    (observation,) = step_observations(records, batching=Batching.STEP, batch_units=32)

    assert observation.failures == 1
    assert observation.clean_count == 2 and observation.clean_mean == 0.5


def test_a_record_without_an_integer_step_tag_is_rejected_in_step_mode() -> None:
    with pytest.raises(IntegrityError, match="u-00000 has no integer 'step' tag"):
        step_observations([make_unit_record(0, step=None)], batching=Batching.STEP, batch_units=32)


def failing(groups_failed: list[int], group_size: int = 4, groups: int = 8) -> list[Record]:
    records, index = [], 0
    for group in range(groups):
        for position in range(group_size):
            failed = position < groups_failed[group]
            kind = FailureKind.SANDBOX if failed else None
            action = Action.MASK if failed else Action.SCORE
            records.append(
                make_unit_record(
                    index, failure_kind=kind, action=action, reward=1.0, step=0, group=f"g{group}"
                )
            )
            index += 1
    return records


def test_the_design_effect_is_near_one_for_spread_failures_and_the_group_size_for_a_burst() -> None:
    (spread,) = step_observations(
        failing([1, 1, 1, 1, 1, 1, 1, 1]), batching=Batching.STEP, batch_units=32
    )
    (burst,) = step_observations(
        failing([4, 0, 0, 0, 0, 0, 0, 0]), batching=Batching.STEP, batch_units=32
    )
    (none,) = step_observations(failing([0] * 8), batching=Batching.STEP, batch_units=32)

    assert design_effect(spread) == 1.0
    assert design_effect(burst) == 4.0
    assert design_effect(none) == 1.0
