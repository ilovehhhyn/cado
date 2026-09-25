"""Every domain value either satisfies the record contract at construction or is rejected with a message that names the rule."""

from __future__ import annotations

import pytest

from tests.support import make_failure, make_record
from tplane.schema import (
    KIND_CLASS,
    SCHEMA_VERSION,
    Action,
    CostLine,
    Decision,
    ExecResult,
    FailureClass,
    FailureKind,
    MismatchSummary,
    Outcome,
    ResourceSummary,
    SchemaError,
)


def test_every_failure_kind_maps_to_exactly_one_class() -> None:
    assert set(KIND_CLASS) == set(FailureKind)
    assert set(KIND_CLASS.values()) == set(FailureClass)


def test_failure_class_is_derived_from_kind() -> None:
    failure = make_failure(kind=FailureKind.OOM_GPU)

    assert failure.failure_class is FailureClass.INFRA


def test_failure_rejects_unsorted_evidence() -> None:
    with pytest.raises(SchemaError, match="evidence must be sorted and unique"):
        make_failure(evidence=("b=1", "a=1"))


def test_failure_rejects_duplicate_evidence() -> None:
    with pytest.raises(SchemaError, match="evidence must be sorted and unique"):
        make_failure(evidence=("a=1", "a=1"))


def test_failure_rejects_empty_message() -> None:
    with pytest.raises(SchemaError, match="message must not be empty"):
        make_failure(message="")


def test_resource_summary_rejects_mismatched_device_lists() -> None:
    with pytest.raises(SchemaError, match="both must list every sampled device"):
        ResourceSummary(
            sample_count=1, peak_host_rss_bytes=1, peak_gpu_used_bytes=(1,), gpu_total_bytes=(1, 2)
        )


def test_exec_result_requires_exactly_one_of_exit_code_and_signal() -> None:
    for exit_code, signal in ((None, None), (0, 9)):
        with pytest.raises(SchemaError, match="exactly one of exit_code and signal"):
            ExecResult(
                exit_code=exit_code,
                signal=signal,
                timed_out=False,
                elapsed_ms=1,
                stdout_tail="",
                stderr_tail="",
            )


def test_record_requires_failure_exactly_when_outcome_is_failed() -> None:
    with pytest.raises(SchemaError, match="outcome is failed exactly when failure is present"):
        make_record(outcome=Outcome.FAILED, failure=None)
    with pytest.raises(SchemaError, match="outcome is failed exactly when failure is present"):
        make_record(outcome=Outcome.OK, failure=make_failure())


def test_record_rejects_end_before_start() -> None:
    with pytest.raises(SchemaError, match="ended_at_ms must be at or after started_at_ms"):
        make_record(started_at_ms=10, ended_at_ms=9)


def test_record_rejects_identifier_with_path_separator() -> None:
    with pytest.raises(SchemaError, match="unit_id must match"):
        make_record(unit_id="../escape")


def test_record_rejects_decision_attempt_that_disagrees_with_record_attempt() -> None:
    with pytest.raises(SchemaError, match="decision.attempt must equal attempt"):
        make_record(attempt=2, decision=Decision(action=Action.SCORE, reason="ok", attempt=1))


def test_record_rejects_unsorted_or_duplicate_tags() -> None:
    with pytest.raises(SchemaError, match="tags must be sorted by key and unique"):
        make_record(tags=(("b", "1"), ("a", "1")))
    with pytest.raises(SchemaError, match="tags must be sorted by key and unique"):
        make_record(tags=(("a", "1"), ("a", "2")))


def test_record_rejects_foreign_schema_version() -> None:
    with pytest.raises(SchemaError, match=f"schema_version must be {SCHEMA_VERSION}"):
        make_record(schema_version=SCHEMA_VERSION + 1)


def test_record_rejects_non_finite_reward() -> None:
    for reward in (float("nan"), float("inf")):
        with pytest.raises(SchemaError, match="reward must be finite"):
            make_record(reward=reward)


def test_valid_record_round_trips_its_fields() -> None:
    record = make_record(unit_id="step-00017")

    assert record.unit_id == "step-00017"
    assert record.outcome is Outcome.OK
    assert record.failure is None


def mismatch(**changes: float) -> MismatchSummary:
    fields: dict[str, float] = {
        "kl": 1e-3,
        "kl_standard_error": 1e-4,
        "ratio_mean": 1.0,
        "ratio_standard_error": 1e-3,
    }
    fields.update(changes)
    return MismatchSummary(sequences=4, tokens=40, **fields)


def test_mismatch_summary_rejects_negative_or_non_finite_statistics() -> None:
    for name in ("kl", "kl_standard_error", "ratio_mean", "ratio_standard_error"):
        for value in (-1e-9, float("nan"), float("inf")):
            with pytest.raises(
                SchemaError, match="mismatch statistics must be finite and non-negative"
            ):
                mismatch(**{name: value})


def test_mismatch_summary_rejects_fewer_than_two_sequences_or_fewer_tokens_than_sequences() -> None:
    with pytest.raises(
        SchemaError, match="mismatch needs at least 2 sequences and one token per sequence"
    ):
        MismatchSummary(
            sequences=1,
            tokens=5,
            kl=0.0,
            kl_standard_error=0.0,
            ratio_mean=1.0,
            ratio_standard_error=0.0,
        )
    with pytest.raises(SchemaError, match="got 3 sequences and 2 tokens"):
        MismatchSummary(
            sequences=3,
            tokens=2,
            kl=0.0,
            kl_standard_error=0.0,
            ratio_mean=1.0,
            ratio_standard_error=0.0,
        )


def test_cost_line_rejects_non_finite_quantities() -> None:
    with pytest.raises(SchemaError, match="cost quantities must be finite and non-negative"):
        CostLine(cpu_seconds=float("nan"), gpu_seconds=0.0, tokens_in=0, tokens_out=0, usd=None)


@pytest.mark.parametrize(
    ("outcome", "action", "reward", "message"),
    (
        (Outcome.OK, Action.MASK, None, "decision must be score exactly when outcome is ok"),
        (Outcome.FAILED, Action.SCORE, 1.0, "decision must be score exactly when outcome is ok"),
        (Outcome.FAILED, Action.ZERO, 1.0, "a zero decision requires reward 0.0, got 1.0"),
        (
            Outcome.FAILED,
            Action.MASK,
            0.5,
            "reward must be absent for mask, retry and abort, got 0.5",
        ),
        (Outcome.OK, Action.SCORE, None, "a scored trajectory or eval must have a reward"),
    ),
)
def test_record_rejects_a_decision_that_disagrees_with_its_outcome_or_reward(
    outcome: Outcome, action: Action, reward: float | None, message: str
) -> None:
    failure = None if outcome is Outcome.OK else make_failure()

    with pytest.raises(SchemaError, match=message):
        make_record(
            outcome=outcome,
            failure=failure,
            reward=reward,
            decision=Decision(action=action, reason="x", attempt=1),
        )
