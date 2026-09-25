"""The benchmark hooks apply each fault exactly from its onset step, write ground truth only to the sidecar, and build records the monitor reads."""

from __future__ import annotations

import pytest

from benchmarks.midrun.hooks import (
    Fault,
    FaultSchedule,
    HookError,
    apply_reward_fault,
    parse_schedule,
    trajectory_records,
)
from tplane.schema import Action, FailureKind, Outcome
from tplane.wire import parse_record, render_record

UIDS = ["a"] * 4 + ["b"] * 4
REWARDS = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_the_schedule_is_parsed_from_the_environment_and_rejects_unknown_values() -> None:
    schedule = parse_schedule(
        {
            "FAULT": "sandbox_partly_typed",
            "FAULT_START": "100",
            "FAULT_RATE": "0.3",
            "FAULT_TYPED": "0.3",
        }
    )

    assert schedule == FaultSchedule(
        fault=Fault.SANDBOX_PARTLY_TYPED, start=100, rate=0.3, typed=0.3
    )
    assert parse_schedule({}) == FaultSchedule(fault=Fault.NONE, start=0, rate=0.0, typed=0.0)
    with pytest.raises(HookError, match="FAULT must be one of"):
        parse_schedule({"FAULT": "banana"})
    with pytest.raises(HookError, match="FAULT_RATE must be in \\[0, 1\\], got 1.5"):
        parse_schedule({"FAULT": "sandbox_masked", "FAULT_START": "1", "FAULT_RATE": "1.5"})
    with pytest.raises(HookError, match="sandbox_masked needs FAULT_START and FAULT_RATE"):
        parse_schedule({"FAULT": "sandbox_masked"})


def test_no_fault_before_the_onset_step() -> None:
    schedule = FaultSchedule(fault=Fault.SANDBOX_ZEROED, start=10, rate=1.0, typed=1.0)

    outcome = apply_reward_fault(schedule, step=9, seed=1, rewards=REWARDS, uids=UIDS)

    assert outcome.trainer_rewards == REWARDS
    assert set(outcome.truth) == {"healthy"}


def test_a_zeroed_outage_scores_every_failed_sample_zero_and_types_it() -> None:
    schedule = FaultSchedule(fault=Fault.SANDBOX_ZEROED, start=10, rate=1.0, typed=1.0)

    outcome = apply_reward_fault(schedule, step=10, seed=1, rewards=REWARDS, uids=UIDS)

    assert outcome.trainer_rewards == [0.0] * 8
    assert set(outcome.truth) == {"sandbox_typed"}
    assert set(outcome.actions) == {Action.ZERO}


def test_a_masked_outage_gives_failed_samples_their_group_mean_so_their_advantage_is_zero() -> None:
    schedule = FaultSchedule(fault=Fault.SANDBOX_MASKED, start=0, rate=0.5, typed=1.0)

    outcome = apply_reward_fault(schedule, step=0, seed=3, rewards=REWARDS, uids=UIDS)

    failed = [index for index, truth in enumerate(outcome.truth) if truth == "sandbox_typed"]
    assert len(failed) > 0
    for index in failed:
        same_group = [
            REWARDS[other]
            for other in range(8)
            if UIDS[other] == UIDS[index] and outcome.truth[other] == "healthy"
        ]
        expected = sum(same_group) / len(same_group) if same_group else 0.0
        assert outcome.trainer_rewards[index] == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert outcome.actions[index] is Action.MASK


def test_an_untyped_outage_looks_like_a_wrong_answer_in_the_record() -> None:
    schedule = FaultSchedule(fault=Fault.SANDBOX_UNTYPED, start=0, rate=1.0, typed=0.0)

    outcome = apply_reward_fault(schedule, step=0, seed=1, rewards=REWARDS, uids=UIDS)
    records = trajectory_records(
        outcome, run_id="run-1", step=0, uids=UIDS, started_at_ms=1_000, ended_at_ms=2_000
    )

    assert set(outcome.truth) == {"sandbox_untyped"}
    assert all(record.outcome is Outcome.OK and record.reward == 0.0 for record in records)


def test_the_fault_draw_is_reproducible_for_a_seed_and_step() -> None:
    schedule = FaultSchedule(fault=Fault.SANDBOX_PARTLY_TYPED, start=0, rate=0.5, typed=0.5)

    first = apply_reward_fault(schedule, step=4, seed=7, rewards=REWARDS, uids=UIDS)
    second = apply_reward_fault(schedule, step=4, seed=7, rewards=REWARDS, uids=UIDS)

    assert first == second


def test_trajectory_records_carry_step_group_and_decision_and_round_trip() -> None:
    schedule = FaultSchedule(fault=Fault.GRADER_MISSING, start=0, rate=1.0, typed=1.0)
    outcome = apply_reward_fault(schedule, step=3, seed=1, rewards=REWARDS, uids=UIDS)

    records = trajectory_records(
        outcome, run_id="run-1", step=3, uids=UIDS, started_at_ms=1_000, ended_at_ms=2_000
    )

    assert [dict(record.tags) for record in records][:2] == [
        {"group": "a", "step": "3"},
        {"group": "a", "step": "3"},
    ]
    assert all(
        record.failure is not None and record.failure.kind is FailureKind.MISSING_REWARD
        for record in records
    )
    assert all(parse_record(render_record(record)) == record for record in records)
