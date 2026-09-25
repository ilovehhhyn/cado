"""Builders and fakes shared by the tplane tests; every fake records its calls and rejects the unexpected."""

from __future__ import annotations

import random
from pathlib import Path

from tplane.schema import (
    SCHEMA_VERSION,
    Action,
    CostLine,
    Decision,
    Failure,
    FailureKind,
    MismatchSummary,
    Outcome,
    Record,
    ResourceSummary,
    Stage,
    UnitKind,
)
from tplane.store import StartedMarker, StoreError, finished_message, stale_marker_message


def make_failure(
    *,
    kind: FailureKind = FailureKind.SANDBOX,
    stage: Stage = Stage.ROLLOUT,
    retryable: bool = True,
    message: str = "sandbox: connection refused",
    evidence: tuple[str, ...] = ("exception_type=ConnectionRefusedError",),
) -> Failure:
    """Build a failure whose defaults are valid, so a test changes one field."""
    return Failure(kind=kind, stage=stage, retryable=retryable, message=message, evidence=evidence)


def make_resources(
    *, peak_gpu: tuple[int, ...] = (1024,), total_gpu: tuple[int, ...] = (4096,)
) -> ResourceSummary:
    """Build a one-sample resource summary."""
    return ResourceSummary(
        sample_count=1,
        peak_host_rss_bytes=2048,
        peak_gpu_used_bytes=peak_gpu,
        gpu_total_bytes=total_gpu,
    )


def make_cost() -> CostLine:
    """Build a cost line with no price."""
    return CostLine(cpu_seconds=1.0, gpu_seconds=0.0, tokens_in=0, tokens_out=0, usd=None)


def make_decision(*, action: Action = Action.SCORE, attempt: int = 1) -> Decision:
    """Build a decision for the given attempt."""
    return Decision(action=action, reason="test", attempt=attempt)


def make_record(
    *,
    schema_version: int = SCHEMA_VERSION,
    run_id: str = "run-1",
    unit_id: str = "unit-1",
    kind: UnitKind = UnitKind.TRAJECTORY,
    attempt: int = 1,
    source: str | None = "swe-tasks",
    started_at_ms: int = 1_000,
    ended_at_ms: int = 2_000,
    outcome: Outcome = Outcome.OK,
    failure: Failure | None = None,
    reward: float | None = 1.0,
    resources: ResourceSummary | None = None,
    cost: CostLine | None = None,
    decision: Decision | None = None,
    tags: tuple[tuple[str, str], ...] = (),
    mismatch: MismatchSummary | None = None,
) -> Record:
    """Build a valid successful record; pass one field to make it invalid."""
    return Record(
        schema_version=schema_version,
        run_id=run_id,
        unit_id=unit_id,
        kind=kind,
        attempt=attempt,
        source=source,
        started_at_ms=started_at_ms,
        ended_at_ms=ended_at_ms,
        outcome=outcome,
        failure=failure,
        reward=reward,
        resources=make_resources() if resources is None else resources,
        cost=make_cost() if cost is None else cost,
        decision=make_decision(
            attempt=attempt, action=Action.SCORE if outcome is Outcome.OK else Action.MASK
        )
        if decision is None
        else decision,
        tags=tags,
        mismatch=mismatch,
    )


class MemoryRecordStore:
    """In-memory RecordStore that enforces the same exactly-once rules as the file store."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self.markers: dict[tuple[str, int], int] = {}

    def append(self, record: Record) -> None:
        if any(
            item.unit_id == record.unit_id and item.attempt == record.attempt
            for item in self.records
        ):
            raise StoreError(
                f"record {record.unit_id}--{record.attempt:02d} already exists; a unit attempt is written exactly once"
            )
        self.records.append(record)

    def iterate(self) -> tuple[Record, ...]:
        return tuple(sorted(self.records, key=lambda item: (item.unit_id, item.attempt)))

    def mark_started(self, unit_id: str, attempt: int, at_ms: int) -> None:
        if any(item.unit_id == unit_id and item.attempt == attempt for item in self.records):
            raise StoreError(finished_message(unit_id, attempt))
        if (unit_id, attempt) in self.markers:
            raise StoreError(
                stale_marker_message(
                    unit_id, attempt, Path("started") / f"{unit_id}--{attempt:02d}"
                )
            )
        self.markers[(unit_id, attempt)] = at_ms

    def clear_started(self, unit_id: str, attempt: int) -> None:
        if (unit_id, attempt) not in self.markers:
            raise StoreError(f"start marker {unit_id}--{attempt:02d} does not exist")
        del self.markers[(unit_id, attempt)]

    def started_without_record(self) -> tuple[StartedMarker, ...]:
        written = {(item.unit_id, item.attempt) for item in self.records}
        return tuple(
            StartedMarker(unit_id=key[0], attempt=key[1], at_ms=at_ms)
            for key, at_ms in sorted(self.markers.items())
            if key not in written
        )


def make_unit_record(
    index: int,
    *,
    source: str | None = "swe",
    reward: float | None = 1.0,
    failure_kind: FailureKind | None = None,
    action: Action = Action.SCORE,
    kind: UnitKind = UnitKind.TRAJECTORY,
    mismatch: MismatchSummary | None = None,
    step: int | None = None,
    group: str | None = None,
) -> Record:
    """Build the record of unit `index`, started at index seconds, whose reward follows the decision like reward_for."""
    failure = None if failure_kind is None else make_failure(kind=failure_kind, retryable=False)
    seen_reward = {Action.SCORE: reward, Action.ZERO: 0.0}.get(action)
    return make_record(
        unit_id=f"{'s' if kind is UnitKind.TRAIN_STEP else 'u'}-{index:05d}",
        kind=kind,
        source=source,
        started_at_ms=1_000 * index,
        ended_at_ms=1_000 * index + 500,
        outcome=Outcome.FAILED if failure is not None else Outcome.OK,
        failure=failure,
        reward=seen_reward,
        decision=make_decision(action=action),
        mismatch=mismatch,
        tags=tuple(
            sorted(
                ([("group", group)] if group is not None else [])
                + ([("step", str(step))] if step is not None else [])
            )
        ),
    )


BASELINE_INFRA_RATE = 0.01


def incident_series(
    *,
    seed: int,
    units: int = 400,
    change_at: int = 200,
    pass_rate: float = 0.6,
    pass_rate_after: float | None = None,
    infra_rate_after: float = 0.0,
    detected_fraction: float = 1.0,
    detected_action: Action = Action.MASK,
) -> list[Record]:
    """Build a seeded run of one source: 1% masked sandbox failures, then from change_at a new failure or pass rate.

    After the change a sandbox failure is typed with probability detected_fraction and then gets detected_action;
    an untyped failure looks like a wrong answer (reward 0 with no failure), as in the MiMo incident.
    """
    rng = random.Random(seed)
    records: list[Record] = []
    for index in range(units):
        after = index >= change_at
        infra_rate = infra_rate_after if after else BASELINE_INFRA_RATE
        rate = pass_rate_after if after and pass_rate_after is not None else pass_rate
        if rng.random() < infra_rate:
            if not after or rng.random() < detected_fraction:
                action = detected_action if after else Action.MASK
                records.append(
                    make_unit_record(index, failure_kind=FailureKind.SANDBOX, action=action)
                )
            else:
                records.append(make_unit_record(index, reward=0.0))
            continue
        records.append(make_unit_record(index, reward=1.0 if rng.random() < rate else 0.0))
    return records


def mismatch_series(
    *,
    seed: int,
    steps: int = 100,
    change_at: int | None = None,
    kl_healthy: float = 4e-4,
    kl_after: float = 4e-4,
    ratio_after: float = 1.0,
    ratio_standard_error: float = 2e-3,
    stride: int = 10,
) -> list[Record]:
    """Build seeded train-step records, one per `stride` units, whose kl varies by a factor of about 1.6 between steps."""
    rng = random.Random(seed)
    records: list[Record] = []
    for step in range(steps):
        after = change_at is not None and step >= change_at
        kl = (kl_after if after else kl_healthy) * 10 ** rng.gauss(0.0, 0.2)
        ratio = rng.gauss(ratio_after if after else 1.0, ratio_standard_error)
        summary = MismatchSummary(
            sequences=512,
            tokens=512 * 400,
            kl=kl,
            kl_standard_error=0.05 * kl,
            ratio_mean=ratio,
            ratio_standard_error=ratio_standard_error,
        )
        records.append(
            make_unit_record(
                10 * step + 5, source=None, reward=None, kind=UnitKind.TRAIN_STEP, mismatch=summary
            )
        )
    return records


def batched_series(
    *,
    seed: int,
    steps: int = 300,
    change_at: int = 150,
    groups: int = 8,
    rollouts: int = 4,
    pass_rate: float = 0.5,
    learning_gain: float = 0.1,
    group_spread: float = 0.25,
    pass_rate_after: float | None = None,
    infra_rate_after: float = 0.0,
    detected_fraction: float = 1.0,
    detected_action: Action = Action.MASK,
    burst_every: int = 0,
) -> list[Record]:
    """Build a seeded batched RL run: each step has `groups` prompts of `rollouts` rollouts that share a difficulty.

    The healthy pass rate rises by learning_gain over the run, and each prompt's own rate is drawn around it with
    sd group_spread, so rollouts of one prompt are correlated, as in GRPO. From change_at, sandbox failures hit
    each rollout with probability infra_rate_after; a failure is typed with probability detected_fraction and then
    gets detected_action, and an untyped failure looks like a wrong answer. burst_every > 0 makes one whole prompt
    group fail with a typed, masked sandbox error every burst_every healthy steps.
    """
    rng = random.Random(seed)
    records: list[Record] = []
    index = 0
    for step in range(steps):
        after = step >= change_at
        base = pass_rate + learning_gain * step / steps
        if after and pass_rate_after is not None:
            base = pass_rate_after + learning_gain * step / steps
        burst_group = (
            rng.randrange(groups)
            if burst_every > 0 and step % burst_every == 0 and not after
            else None
        )
        for group in range(groups):
            difficulty = min(1.0, max(0.0, rng.gauss(base, group_spread)))
            for _ in range(rollouts):
                tag = {"step": step, "group": f"g{step}-{group}"}
                if group == burst_group:
                    records.append(
                        make_unit_record(
                            index, failure_kind=FailureKind.SANDBOX, action=Action.MASK, **tag
                        )
                    )
                elif after and rng.random() < infra_rate_after:
                    if rng.random() < detected_fraction:
                        records.append(
                            make_unit_record(
                                index,
                                failure_kind=FailureKind.SANDBOX,
                                action=detected_action,
                                **tag,
                            )
                        )
                    else:
                        records.append(make_unit_record(index, reward=0.0, **tag))
                else:
                    records.append(
                        make_unit_record(
                            index, reward=1.0 if rng.random() < difficulty else 0.0, **tag
                        )
                    )
                index += 1
    return records
