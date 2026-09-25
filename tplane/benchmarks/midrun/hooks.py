"""Inject one scheduled fault into a verl GRPO step and build the tplane records the monitor reads.

The patched trainer calls apply_reward_fault after the grader and before advantages. It calls trajectory_records
and train_step_record, then write_step, once per step. Ground truth goes only to the sidecar truth file, never into
a record, so the monitor cannot see the label. A masked failure gets the mean clean reward of its prompt group, so
its GRPO advantage is zero and it cannot move the policy or the group baseline.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tplane.files import publish_new_file, write_text
from tplane.mismatch import SequenceSums, summarise_mismatch
from tplane.schema import (
    SCHEMA_VERSION,
    Action,
    CostLine,
    Decision,
    Failure,
    FailureKind,
    Outcome,
    Record,
    ResourceSummary,
    Stage,
    UnitKind,
)
from tplane.wire import render_record

SOURCE: Final[str] = "leetcode_rh"
NO_RESOURCES: Final[ResourceSummary] = ResourceSummary(
    sample_count=0, peak_host_rss_bytes=0, peak_gpu_used_bytes=(), gpu_total_bytes=()
)
NO_COST: Final[CostLine] = CostLine(
    cpu_seconds=0.0, gpu_seconds=0.0, tokens_in=0, tokens_out=0, usd=None
)


class HookError(ValueError):
    """The fault schedule is a closed set with an onset step and rates in [0, 1]."""


class Fault(StrEnum):
    """The one fault a benchmark run injects from its onset step."""

    GRADER_MISSING = "grader_missing"
    LR_SPIKE = "lr_spike"
    NONE = "none"
    SANDBOX_MASKED = "sandbox_masked"
    SANDBOX_PARTLY_TYPED = "sandbox_partly_typed"
    SANDBOX_UNTYPED = "sandbox_untyped"
    SANDBOX_ZEROED = "sandbox_zeroed"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"


REWARD_FAULTS: Final[frozenset[Fault]] = frozenset(
    {
        Fault.GRADER_MISSING,
        Fault.SANDBOX_MASKED,
        Fault.SANDBOX_PARTLY_TYPED,
        Fault.SANDBOX_UNTYPED,
        Fault.SANDBOX_ZEROED,
    }
)
TYPED_SHARE: Final[dict[Fault, float | None]] = {
    Fault.GRADER_MISSING: 1.0,
    Fault.SANDBOX_MASKED: 1.0,
    Fault.SANDBOX_PARTLY_TYPED: None,  # from FAULT_TYPED
    Fault.SANDBOX_UNTYPED: 0.0,
    Fault.SANDBOX_ZEROED: 1.0,
}


@dataclass(frozen=True, kw_only=True)
class FaultSchedule:
    """Which fault starts at which step, the share of samples it hits, and the share of those that are typed."""

    fault: Fault
    start: int
    rate: float
    typed: float


@dataclass(frozen=True, kw_only=True)
class RewardOutcome:
    """Per sample: the reward the trainer uses, the record's decision and reward, and the hidden truth."""

    trainer_rewards: list[float]
    actions: list[Action]
    record_rewards: list[float | None]
    failure_kinds: list[FailureKind | None]
    truth: list[str]


def parse_schedule(environment: Mapping[str, str]) -> FaultSchedule:
    """Parse FAULT, FAULT_START, FAULT_RATE and FAULT_TYPED; an absent FAULT means no fault."""
    name = environment.get("FAULT", Fault.NONE.value)
    if name not in {member.value for member in Fault}:
        raise HookError(
            f"FAULT must be one of {sorted(member.value for member in Fault)}, got {name!r}"
        )
    fault = Fault(name)
    if fault is Fault.NONE:
        return FaultSchedule(fault=fault, start=0, rate=0.0, typed=0.0)
    if "FAULT_START" not in environment or "FAULT_RATE" not in environment:
        raise HookError(f"{fault.value} needs FAULT_START and FAULT_RATE")
    rate = float(environment["FAULT_RATE"])
    typed = float(environment.get("FAULT_TYPED", "0"))
    if fault is not Fault.LR_SPIKE and fault is not Fault.TEMPERATURE and fault is not Fault.TOP_P:
        if not 0 <= rate <= 1:
            raise HookError(f"FAULT_RATE must be in [0, 1], got {rate}")
    if not 0 <= typed <= 1:
        raise HookError(f"FAULT_TYPED must be in [0, 1], got {typed}")
    return FaultSchedule(fault=fault, start=int(environment["FAULT_START"]), rate=rate, typed=typed)


def rollout_overrides(schedule: FaultSchedule, *, step: int) -> dict[str, float]:
    """Return the vLLM sampling parameters to override at this step; the trainer's log-probs keep the config."""
    if step < schedule.start:
        return {}
    if schedule.fault is Fault.TOP_P:
        return {"top_p": schedule.rate}
    if schedule.fault is Fault.TEMPERATURE:
        return {"temperature": schedule.rate}
    return {}


def learning_rate_scale(schedule: FaultSchedule, *, step: int) -> float:
    """Return the factor to multiply the learning rate by at this step."""
    return schedule.rate if schedule.fault is Fault.LR_SPIKE and step >= schedule.start else 1.0


def apply_reward_fault(
    schedule: FaultSchedule, *, step: int, seed: int, rewards: Sequence[float], uids: Sequence[str]
) -> RewardOutcome:
    """Draw which samples the fault hits at this step, reproducibly from (seed, step), and what each one records."""
    count = len(rewards)
    active = schedule.fault in REWARD_FAULTS and step >= schedule.start
    rng = random.Random(f"tplane-benchmark:{seed}:{step}")
    share = TYPED_SHARE.get(schedule.fault)
    typed_share = schedule.typed if share is None else share
    hit = [active and rng.random() < schedule.rate for _ in range(count)]
    typed = [flag and rng.random() < typed_share for flag in hit]
    truth = [
        _truth(schedule.fault, flag, is_typed) for flag, is_typed in zip(hit, typed, strict=True)
    ]
    group_means = _clean_group_means(rewards, uids, hit)
    trainer, actions, recorded, kinds = [], [], [], []
    for index in range(count):
        action, kind = _decision(schedule.fault, hit[index], typed[index])
        actions.append(action)
        kinds.append(kind)
        if not hit[index]:
            trainer.append(rewards[index])
            recorded.append(rewards[index])
        elif action is Action.MASK:
            trainer.append(group_means.get(uids[index], 0.0))
            recorded.append(None)
        else:
            trainer.append(0.0)
            recorded.append(0.0)
    return RewardOutcome(
        trainer_rewards=trainer,
        actions=actions,
        record_rewards=recorded,
        failure_kinds=kinds,
        truth=truth,
    )


def trajectory_records(
    outcome: RewardOutcome,
    *,
    run_id: str,
    step: int,
    uids: Sequence[str],
    started_at_ms: int,
    ended_at_ms: int,
) -> list[Record]:
    """Build one trajectory record per sample, tagged with its step and prompt group."""
    records = []
    for index, uid in enumerate(uids):
        kind = outcome.failure_kinds[index]
        failure = None
        if kind is not None:
            failure = Failure(
                kind=kind,
                stage=Stage.GRADE,
                retryable=False,
                message=f"{kind.value}: injected by the benchmark",
                evidence=("injected=true",),
            )
        action = outcome.actions[index]
        records.append(
            Record(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                unit_id=f"s{step:05d}-{index:04d}",
                kind=UnitKind.TRAJECTORY,
                attempt=1,
                source=SOURCE,
                started_at_ms=started_at_ms,
                ended_at_ms=ended_at_ms,
                outcome=Outcome.OK if failure is None else Outcome.FAILED,
                failure=failure,
                reward=outcome.record_rewards[index],
                resources=NO_RESOURCES,
                cost=NO_COST,
                decision=Decision(action=action, reason="benchmark", attempt=1),
                tags=(("group", uid), ("step", str(step))),
            )
        )
    return records


def train_step_record(
    *, run_id: str, step: int, sums: Sequence[SequenceSums], started_at_ms: int, ended_at_ms: int
) -> Record:
    """Build the train-step record with the batch's train-inference mismatch summary."""
    return Record(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        unit_id=f"step-{step:05d}",
        kind=UnitKind.TRAIN_STEP,
        attempt=1,
        source=None,
        started_at_ms=started_at_ms,
        ended_at_ms=ended_at_ms,
        outcome=Outcome.OK,
        failure=None,
        reward=None,
        resources=NO_RESOURCES,
        cost=NO_COST,
        decision=Decision(action=Action.SCORE, reason="benchmark", attempt=1),
        tags=(("step", str(step)),),
        mismatch=summarise_mismatch(sums),
    )


def write_step(
    directory: Path, *, step: int, records: Sequence[Record], outcome: RewardOutcome
) -> None:
    """Publish this step's records and its hidden truth as two new JSONL files."""
    directory.mkdir(parents=True, exist_ok=True)
    record_lines = "".join(
        json.dumps(render_record(record), sort_keys=True) + "\n" for record in records
    )
    truth_lines = "".join(
        json.dumps({"step": step, "index": index, "truth": truth}) + "\n"
        for index, truth in enumerate(outcome.truth)
    )
    publish_new_file(
        directory / f"records-{step:05d}.jsonl",
        write_text(record_lines),
        exists_message=f"step {step} records exist",
    )
    publish_new_file(
        directory / f"truth-{step:05d}.jsonl",
        write_text(truth_lines),
        exists_message=f"step {step} truth exists",
    )


def _truth(fault: Fault, hit: bool, typed: bool) -> str:
    if not hit:
        return "healthy"
    if fault is Fault.GRADER_MISSING:
        return "grader_missing"
    return "sandbox_typed" if typed else "sandbox_untyped"


def _decision(fault: Fault, hit: bool, typed: bool) -> tuple[Action, FailureKind | None]:
    if not hit or not typed:
        return Action.SCORE, None
    if fault is Fault.GRADER_MISSING:
        return Action.MASK, FailureKind.MISSING_REWARD
    if fault is Fault.SANDBOX_ZEROED:
        return Action.ZERO, FailureKind.SANDBOX
    return Action.MASK, FailureKind.SANDBOX


def _clean_group_means(
    rewards: Sequence[float], uids: Sequence[str], hit: Sequence[bool]
) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for reward, uid, failed in zip(rewards, uids, hit, strict=True):
        if not failed:
            groups.setdefault(uid, []).append(reward)
    return {uid: statistics.fmean(values) for uid, values in groups.items()}
