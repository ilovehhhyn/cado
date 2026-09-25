"""Assemble the persisted record of one finished unit attempt from its measured parts, without effects."""

from __future__ import annotations

from typing import Final, assert_never

from tplane.cost import PriceTable, cost_line
from tplane.schema import (
    SCHEMA_VERSION,
    Action,
    Decision,
    Failure,
    MismatchSummary,
    Outcome,
    Record,
    ResourceSummary,
    UnitKind,
)

MILLISECONDS_PER_SECOND: Final[int] = 1000


def build_record(
    *,
    run_id: str,
    unit_id: str,
    kind: UnitKind,
    attempt: int,
    source: str | None,
    started_at_ms: int,
    ended_at_ms: int,
    failure: Failure | None,
    reward: float | None,
    resources: ResourceSummary,
    decision: Decision,
    tokens_in: int,
    tokens_out: int,
    prices: PriceTable,
    cpus_held: int,
    gpus_held: int,
    tags: tuple[tuple[str, str], ...],
    mismatch: MismatchSummary | None = None,
) -> Record:
    """Assemble the record; cost = elapsed seconds times cpus_held and gpus_held, priced by prices."""
    elapsed_s = (ended_at_ms - started_at_ms) / MILLISECONDS_PER_SECOND
    cost = cost_line(
        cpu_seconds=elapsed_s * cpus_held,
        gpu_seconds=elapsed_s * gpus_held,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        prices=prices,
    )
    return Record(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        unit_id=unit_id,
        kind=kind,
        attempt=attempt,
        source=source,
        started_at_ms=started_at_ms,
        ended_at_ms=ended_at_ms,
        outcome=Outcome.OK if failure is None else Outcome.FAILED,
        failure=failure,
        reward=reward_for(decision, reward),
        resources=resources,
        cost=cost,
        decision=decision,
        tags=tags,
        mismatch=mismatch,
    )


def reward_for(decision: Decision, reward: float | None) -> float | None:
    """Return the reward the trainer sees: the unit's own on score, 0.0 on zero, absent otherwise."""
    match decision.action:
        case Action.SCORE:
            return reward
        case Action.ZERO:
            return 0.0
        case Action.MASK | Action.RETRY | Action.ABORT:
            return None
        case _:
            assert_never(decision.action)


def render_log_line(record: Record) -> str:
    """Render the one-line log entry for a record."""
    failure = record.failure
    cause = (
        "-"
        if failure is None
        else f"{failure.failure_class.value}/{failure.kind.value}@{failure.stage.value}"
    )
    return f"tplane {record.unit_id} attempt {record.attempt} {record.outcome.value} {record.decision.action.value} {cause}"
