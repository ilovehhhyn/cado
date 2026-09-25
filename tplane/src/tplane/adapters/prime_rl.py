"""Record verifiers traces and prime-rl dispatch failures as tplane units, inside the orchestrator before failures are dropped.

prime-rl drops an errored trace before the trainer sees it (algo/base.py iter_trainable_traces), so the adapter must
run where the trace is still visible. Field names follow verifiers v0.3.1 (verifiers/v1/trace.py) and prime-rl
1556db6 (src/prime_rl/orchestrator/types.py). A trace that is ok but whose reward never ran is a grader failure,
never a reward of 0.0 (verifiers issue #2660).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from tplane.adapters.outcome import AdapterError
from tplane.classify import classify_error_name
from tplane.schema import Action, Failure, FailureClass, FailureKind, Record, Stage, UnitKind
from tplane.unit import Run, UnitResult

DISPATCH_KINDS: Final[dict[str, UnitKind]] = {"eval": UnitKind.EVAL, "train": UnitKind.TRAJECTORY}
TRAINABLE_ACTIONS: Final[frozenset[Action]] = frozenset({Action.SCORE, Action.ZERO})


class VerifiersError(Protocol):
    """The fields of verifiers.v1.trace.Error that classification reads."""

    @property
    def type(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def status_code(self) -> int | None: ...


class VerifiersReward(Protocol):
    """verifiers.v1.trace.Reward."""

    @property
    def score(self) -> float: ...

    @property
    def weight(self) -> float: ...


class VerifiersTrace(Protocol):
    """The fields of verifiers.v1.trace.Trace the adapter reads; a reward of None means scoring did not run."""

    @property
    def ok(self) -> bool: ...

    @property
    def errors(self) -> Sequence[VerifiersError]: ...

    @property
    def rewards(self) -> Mapping[str, VerifiersReward | None]: ...


class DispatchFailure(Protocol):
    """The fields of prime_rl.orchestrator.types.DispatchFailure the adapter reads."""

    @property
    def kind(self) -> str: ...

    @property
    def env_name(self) -> str: ...

    @property
    def step(self) -> int: ...

    @property
    def policy_version(self) -> int: ...

    @property
    def task_key(self) -> str: ...

    @property
    def error(self) -> VerifiersError: ...


def failure_from_error(error: VerifiersError, *, stage: Stage) -> Failure:
    """Classify a verifiers error record; a grader failure is attributed to the grade stage."""
    failure = classify_error_name(
        error.type, error.message, stage=stage, status_code=error.status_code
    )
    if failure.failure_class is FailureClass.GRADER:
        return dataclasses.replace(failure, stage=Stage.GRADE)
    return failure


def record_trace(
    run: Run,
    trace: VerifiersTrace,
    *,
    unit_id: str,
    source: str | None,
    kind: UnitKind = UnitKind.TRAJECTORY,
    attempt: int = 1,
) -> UnitResult:
    """Write the record of one finished trace: its weighted reward, or the failure of its last error.

    When the record's decision is retry, the caller must dispatch the task again and record it with attempt + 1;
    the unit has no final record until then, and is_trainable is False for the retry record.
    """
    outcome = _trace_outcome(trace, unit_id)
    with run.unit(unit_id, kind=kind, source=source, attempt=attempt) as unit:
        if isinstance(outcome, Failure):
            unit.fail(outcome)
        unit.reward(outcome)
    return unit.result


def record_dispatch_failure(
    run: Run, failure: DispatchFailure, *, unit_id: str, attempt: int = 1
) -> UnitResult:
    """Write the record of an environment request that failed before it produced an episode."""
    kind = DISPATCH_KINDS.get(failure.kind)
    if kind is None:
        raise AdapterError(
            f"DispatchFailure.kind must be one of {sorted(DISPATCH_KINDS)}, got {failure.kind!r}"
        )
    classified = failure_from_error(failure.error, stage=Stage.SETUP)
    with run.unit(unit_id, kind=kind, source=failure.env_name, attempt=attempt) as unit:
        unit.tag("policy_version", str(failure.policy_version))
        unit.tag("step", str(failure.step))
        unit.tag("task_key", failure.task_key)
        unit.fail(classified)
    return unit.result


def is_trainable(record: Record) -> bool:
    """Return whether the trainer should keep the sample: scored, or zeroed for an agent or timeout failure."""
    return record.decision.action in TRAINABLE_ACTIONS


def _trace_outcome(trace: VerifiersTrace, unit_id: str) -> float | Failure:
    if not trace.ok:
        if len(trace.errors) == 0:
            raise AdapterError(
                f"trace {unit_id} is not ok but has no errors; verifiers records every error"
            )
        return failure_from_error(trace.errors[-1], stage=Stage.ROLLOUT)
    rewards = list(trace.rewards.values())
    scored = [reward for reward in rewards if reward is not None]
    if len(rewards) == 0 or len(scored) != len(rewards):
        return Failure(
            kind=FailureKind.MISSING_REWARD,
            stage=Stage.GRADE,
            retryable=False,
            message=f"missing_reward: trace is ok but {len(rewards) - len(scored)} of {len(rewards)} rewards did not run",
            evidence=(
                f"rewards_missing={len(rewards) - len(scored)}",
                f"rewards_total={len(rewards)}",
            ),
        )
    return math.fsum(reward.score * reward.weight for reward in scored)
