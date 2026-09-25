"""A verifiers trace or prime-rl dispatch failure becomes one tplane record: a missing reward is a grader failure, HTTP 5xx is infrastructure, and only scored or zeroed records are trainable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from tests.support import MemoryRecordStore
from tplane.adapters.outcome import AdapterError
from tplane.adapters.prime_rl import (
    failure_from_error,
    is_trainable,
    record_dispatch_failure,
    record_trace,
)
from tplane.policy import DEFAULT_POLICY, Policy
from tplane.schema import Action, FailureClass, FailureKind, Stage, UnitKind
from tplane.unit import Run, RunOptions


@dataclass(frozen=True)
class Error:
    """verifiers.v1.trace.Error at v0.3.1."""

    type: str
    message: str
    status_code: int | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class Reward:
    """verifiers.v1.trace.Reward at v0.3.1."""

    score: float
    weight: float = 1.0


@dataclass(frozen=True)
class Trace:
    """The verifiers.v1.trace.Trace fields the adapter reads."""

    ok: bool
    errors: Sequence[Error] = ()
    rewards: Mapping[str, Reward | None] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchFailure:
    """prime_rl.orchestrator.types.DispatchFailure at 1556db6."""

    kind: str
    env_name: str
    group_id: str
    step: int
    policy_version: int
    task_type: str
    task_key: str
    task_hash: str
    error: Error


def make_run(policy: Policy = DEFAULT_POLICY) -> tuple[Run, MemoryRecordStore]:
    store = MemoryRecordStore()
    run = Run(
        run_id="run-1",
        store=store,
        options=RunOptions(policy=policy, sample_interval_ms=10),
        read_host=lambda: 1,
        read_gpu=lambda: (),
        gpu_total_bytes=(),
    )
    return run, store


def dispatch(error: Error, *, kind: str = "train") -> DispatchFailure:
    return DispatchFailure(
        kind=kind,
        env_name="swe",
        group_id="g-7",
        step=12,
        policy_version=11,
        task_type="swe",
        task_key="repo-42",
        task_hash="ab12",
        error=error,
    )


def test_an_ok_trace_scores_the_weighted_reward_sum() -> None:
    run, store = make_run()
    trace = Trace(
        ok=True, rewards={"pass": Reward(score=1.0, weight=0.5), "format": Reward(score=0.2)}
    )

    result = record_trace(run, trace, unit_id="t-1", source="swe")

    assert result.record.decision.action is Action.SCORE
    assert result.record.reward == pytest.approx(0.7, rel=1e-12, abs=0.0)
    assert store.iterate() == (result.record,)


def test_an_ok_trace_whose_scoring_did_not_run_is_a_missing_reward_not_a_zero() -> None:
    # Invariant: verifiers issue #2660 cannot recur; an unread reward file never becomes solved = 0.0.
    run, _ = make_run()
    trace = Trace(ok=True, rewards={"solved": None})

    result = record_trace(run, trace, unit_id="t-1", source="swe")

    assert result.record.failure is not None
    assert (result.record.failure.kind, result.record.failure.stage) == (
        FailureKind.MISSING_REWARD,
        Stage.GRADE,
    )
    assert result.record.decision.action is Action.MASK
    assert result.record.reward is None


def test_an_ok_trace_with_no_rewards_at_all_is_a_missing_reward() -> None:
    run, _ = make_run()

    result = record_trace(run, Trace(ok=True), unit_id="t-1", source="swe")

    assert (
        result.record.failure is not None
        and result.record.failure.kind is FailureKind.MISSING_REWARD
    )


@pytest.mark.parametrize(
    ("error", "kind", "action"),
    (
        (Error(type="SandboxError", message="sandbox gone"), FailureKind.SANDBOX, Action.MASK),
        (
            Error(type="TaskError", message="reward.txt is empty"),
            FailureKind.GRADER_EXCEPTION,
            Action.MASK,
        ),
        (
            Error(type="HarnessError", message="agent crashed"),
            FailureKind.AGENT_EXCEPTION,
            Action.ZERO,
        ),
    ),
)
def test_a_failed_trace_is_classified_by_its_last_error(
    error: Error, kind: FailureKind, action: Action
) -> None:
    run, _ = make_run()
    trace = Trace(ok=False, errors=(Error(type="ToolError", message="earlier"), error))

    result = record_trace(run, trace, unit_id="t-1", source="swe")

    assert result.record.failure is not None
    assert result.record.failure.kind is kind
    assert result.record.decision.action is action


def test_a_failed_trace_without_errors_is_rejected() -> None:
    run, store = make_run()

    with pytest.raises(
        AdapterError, match="trace t-1 is not ok but has no errors; verifiers records every error"
    ):
        record_trace(run, Trace(ok=False), unit_id="t-1", source="swe")

    assert store.records == [] and store.markers == {}


@pytest.mark.parametrize(
    ("error", "kind"),
    (
        (
            Error(type="ProviderError", message="bad gateway", status_code=502),
            FailureKind.ENGINE_CRASH,
        ),
        (Error(type="APIError", message="upstream", status_code=503), FailureKind.ENGINE_CRASH),
        (Error(type="APIError", message="rate limited", status_code=429), FailureKind.ENGINE_CRASH),
        (
            Error(type="APIError", message="bad request", status_code=400),
            FailureKind.AGENT_EXCEPTION,
        ),
        (
            Error(type="OverlongPromptError", message="context too long", status_code=400),
            FailureKind.AGENT_EXCEPTION,
        ),
    ),
)
def test_http_status_decides_between_infrastructure_and_agent_for_unknown_names(
    error: Error, kind: FailureKind
) -> None:
    failure = failure_from_error(error, stage=Stage.ROLLOUT)

    assert failure.kind is kind
    assert f"status_code={error.status_code}" in failure.evidence


def test_a_dispatch_failure_becomes_a_tagged_record_of_its_kind() -> None:
    run, _ = make_run()

    result = record_dispatch_failure(
        run, dispatch(Error(type="SandboxError", message="refused"), kind="eval"), unit_id="g-7"
    )

    record = result.record
    assert record.kind is UnitKind.EVAL
    assert record.source == "swe"
    assert record.failure is not None and record.failure.failure_class is FailureClass.INFRA
    assert dict(record.tags) == {"policy_version": "11", "step": "12", "task_key": "repo-42"}


def test_a_dispatch_failure_of_an_unknown_kind_is_rejected_before_any_record() -> None:
    run, store = make_run()

    with pytest.raises(
        AdapterError,
        match="DispatchFailure.kind must be one of \\['eval', 'train'\\], got 'banana'",
    ):
        record_dispatch_failure(
            run, dispatch(Error(type="SandboxError", message="x"), kind="banana"), unit_id="g-7"
        )

    assert store.records == [] and store.markers == {}


def test_only_scored_or_zeroed_records_are_trainable() -> None:
    run, _ = make_run()
    scored = record_trace(
        run, Trace(ok=True, rewards={"r": Reward(score=1.0)}), unit_id="a", source="swe"
    )
    zeroed = record_trace(
        run,
        Trace(ok=False, errors=(Error(type="HarnessError", message="x"),)),
        unit_id="b",
        source="swe",
    )
    masked = record_trace(run, Trace(ok=True), unit_id="c", source="swe")

    assert [is_trainable(result.record) for result in (scored, zeroed, masked)] == [
        True,
        True,
        False,
    ]


def test_a_retryable_failure_with_attempts_left_is_decided_retry_and_is_not_trainable() -> None:
    run, _ = make_run(Policy(max_attempts=2))
    trace = Trace(ok=False, errors=(Error(type="SandboxError", message="sandbox gone"),))

    first = record_trace(run, trace, unit_id="t-1", source="swe", attempt=1)
    second = record_trace(run, trace, unit_id="t-1", source="swe", attempt=2)

    assert [first.record.decision.action, second.record.decision.action] == [
        Action.RETRY,
        Action.MASK,
    ]
    assert is_trainable(first.record) is False
