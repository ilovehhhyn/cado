"""On batched runs with correlated prompt groups and learning drift, each signal alarms on its planted change, stays silent on healthy training, and attributes reward drops to configuration only with configuration evidence."""

from __future__ import annotations

import math

import pytest

from tests.support import batched_series, make_unit_record, mismatch_series
from tplane.integrity.metrics import IntegrityError
from tplane.integrity.monitor import (
    DEFAULT_INTEGRITY_OPTIONS,
    Alarm,
    IntegrityOptions,
    Signal,
    Verdict,
    detect_alarms,
    mismatch_llr,
)
from tplane.integrity.steps import Batching
from tplane.schema import Action, MismatchSummary

SEEDS = (0, 1, 2)
CHANGE_AT = 150
UNITS_PER_STEP = 32


def only(alarms: tuple[Alarm, ...], signal: Signal) -> list[Alarm]:
    return [alarm for alarm in alarms if alarm.signal is signal]


def step_of(alarm: Alarm) -> int:
    return int(alarm.unit_id.split("-")[1]) // UNITS_PER_STEP


@pytest.mark.parametrize("seed", range(10))
def test_a_healthy_run_with_correlated_prompts_and_learning_drift_raises_no_alarm(
    seed: int,
) -> None:
    # Witness: the per-rollout model raised a false reward alarm on 8 of these 10 runs.
    records = batched_series(seed=seed, change_at=10**9) + mismatch_series(
        seed=seed, steps=300, stride=UNITS_PER_STEP
    )

    assert detect_alarms(records) == ()


@pytest.mark.parametrize("seed", range(5))
def test_typed_failures_confined_to_one_prompt_group_raise_no_failure_rate_alarm(seed: int) -> None:
    # One prompt whose tests all hit a sandbox error is one event, not `rollouts` independent failures.
    records = batched_series(seed=seed, change_at=10**9, burst_every=4)

    assert only(detect_alarms(records), Signal.FAILURE_RATE) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_a_zeroed_sandbox_outage_raises_poisoned_at_once_and_failure_rate_within_five_steps(
    seed: int,
) -> None:
    records = batched_series(seed=seed, infra_rate_after=0.5, detected_action=Action.ZERO)
    first_zeroed = next(
        record for record in records if record.failure is not None and record.reward == 0.0
    )

    alarms = detect_alarms(records)

    (poisoned,) = only(alarms, Signal.POISONED)
    (failure_rate,) = only(alarms, Signal.FAILURE_RATE)
    assert poisoned.unit_id == first_zeroed.unit_id
    assert CHANGE_AT <= step_of(failure_rate) < CHANGE_AT + 5, failure_rate
    assert only(alarms, Signal.REWARD) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_a_masked_sandbox_outage_raises_only_the_failure_rate_alarm(seed: int) -> None:
    alarms = detect_alarms(
        batched_series(seed=seed, infra_rate_after=0.5, detected_action=Action.MASK)
    )

    assert [alarm.signal for alarm in alarms] == [Signal.FAILURE_RATE]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_reward_drop_from_mostly_untyped_sandbox_failures_is_attributed_to_configuration(
    seed: int,
) -> None:
    records = batched_series(seed=seed, infra_rate_after=0.5, detected_fraction=0.3)

    (reward,) = only(detect_alarms(records), Signal.REWARD)

    assert reward.attribution is not None
    assert reward.attribution.verdict is Verdict.CONFIGURATION
    assert step_of(reward) >= CHANGE_AT


@pytest.mark.parametrize("seed", SEEDS)
def test_a_learning_collapse_is_attributed_to_the_algorithm(seed: int) -> None:
    records = batched_series(seed=seed, pass_rate_after=0.2) + mismatch_series(
        seed=seed, steps=300, stride=UNITS_PER_STEP
    )

    alarms = detect_alarms(records)

    (reward,) = only(alarms, Signal.REWARD)
    assert reward.attribution is not None and reward.attribution.verdict is Verdict.ALGORITHM
    assert only(alarms, Signal.FAILURE_RATE) == only(alarms, Signal.MISMATCH) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_a_reward_drop_after_a_mismatch_jump_is_attributed_to_configuration(seed: int) -> None:
    records = batched_series(seed=seed, pass_rate_after=0.2) + mismatch_series(
        seed=seed, steps=300, change_at=CHANGE_AT, kl_after=3e-2, stride=UNITS_PER_STEP
    )

    (reward,) = only(detect_alarms(records), Signal.REWARD)

    assert reward.attribution is not None and reward.attribution.verdict is Verdict.CONFIGURATION


@pytest.mark.parametrize("seed", SEEDS)
def test_a_jump_in_mismatch_kl_raises_a_mismatch_alarm_within_five_steps(seed: int) -> None:
    (alarm,) = detect_alarms(mismatch_series(seed=seed, change_at=50, kl_after=3e-2))

    assert alarm.signal is Signal.MISMATCH
    assert 50 <= (int(alarm.unit_id.split("-")[1]) - 5) // 10 < 55, alarm


@pytest.mark.parametrize("seed", SEEDS)
def test_a_mean_ratio_below_one_raises_a_mismatch_alarm_while_kl_looks_healthy(seed: int) -> None:
    (alarm,) = detect_alarms(mismatch_series(seed=seed, change_at=50, ratio_after=0.97))

    assert alarm.signal is Signal.MISMATCH


def test_untagged_trajectories_are_rejected_in_step_mode_with_the_fix_named() -> None:
    with pytest.raises(
        IntegrityError,
        match="u-00000 has no integer 'step' tag; tag each trajectory with its training step",
    ):
        detect_alarms([make_unit_record(0)])


def test_unit_batching_accepts_untagged_records_and_counts_windows() -> None:
    options = IntegrityOptions(batching=Batching.UNITS, batch_units=32)
    records = [make_unit_record(index, reward=float(index % 2)) for index in range(640)]

    assert detect_alarms(records, options=options) == ()


def test_a_step_with_one_prompt_group_gives_no_reward_evidence_rather_than_a_guess() -> None:
    records = [
        make_unit_record(index, reward=0.0, step=index // 4, group=f"g{index // 4}")
        for index in range(4_000)
    ]

    assert detect_alarms(records) == ()


def test_a_vanishing_ratio_standard_error_gives_a_finite_likelihood_ratio() -> None:
    summary = MismatchSummary(
        sequences=4,
        tokens=40,
        kl=1e-3,
        kl_standard_error=0.0,
        ratio_mean=1.0,
        ratio_standard_error=1e-200,
    )

    assert math.isfinite(mismatch_llr(summary, options=DEFAULT_INTEGRITY_OPTIONS))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"alarm_probability": 0.5}, "alarm_probability must be in \\(0.5, 1\\), got 0.5"),
        ({"hazard": 0.0}, "hazard must be in \\(0, 1\\), got 0.0"),
        (
            {"failure_rate_healthy": 0.2, "failure_rate_broken": 0.1},
            "0 < failure_rate_healthy < failure_rate_broken < 1",
        ),
        ({"healthy_kl": 1e-2, "broken_kl": 1e-3}, "0 < healthy_kl < broken_kl"),
        (
            {"reward_reference_steps": 1},
            "reward_reference_steps must be at least 2 and reward_lag_steps at least 0",
        ),
        (
            {"reward_shift_sd": 0.0},
            "reward_shift_sd, reward_sd_floor, kl_sd_decades and ratio_shift must be positive",
        ),
        ({"batch_units": 1}, "batch_units must be at least 2, got 1"),
        (
            {"reward_scale_steps": 10},
            "reward_scale_steps must be at least reward_reference_steps, got 10 and 20",
        ),
        ({"attribution_steps": -1}, "attribution_steps must be at least 0, got -1"),
        ({"window_units": 0}, "window_units must be at least 1, got 0"),
    ),
)
def test_options_reject_settings_that_cannot_be_honoured(
    changes: dict[str, float], message: str
) -> None:
    with pytest.raises(IntegrityError, match=message):
        IntegrityOptions(**changes)
