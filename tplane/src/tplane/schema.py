"""Immutable domain types for one unit of work and its typed outcome."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

SCHEMA_VERSION: Final[int] = 2  # version 2 adds Record.mismatch
MIN_MISMATCH_SEQUENCES: Final[int] = 2  # a standard error needs at least two clusters
IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class SchemaError(ValueError):
    """A domain value must satisfy the record contract at construction."""


class UnitKind(StrEnum):
    """The kind of work one unit performs."""

    EVAL = "eval"
    TRAIN_STEP = "train_step"
    TRAJECTORY = "trajectory"


class Stage(StrEnum):
    """The phase of a unit in which a failure happened."""

    GRADE = "grade"
    ROLLOUT = "rollout"
    SETUP = "setup"
    TOOL = "tool"
    TRAIN = "train"


class FailureClass(StrEnum):
    """Who is responsible for a failure: the agent, the grader, the infrastructure or the clock."""

    AGENT = "agent"
    GRADER = "grader"
    INFRA = "infra"
    TIMEOUT = "timeout"


class FailureKind(StrEnum):
    """The specific cause of a failure; each kind belongs to exactly one class."""

    AGENT_EXCEPTION = "agent_exception"
    ENGINE_CRASH = "engine_crash"
    GRADER_EXCEPTION = "grader_exception"
    HARDWARE = "hardware"
    MALFORMED_ACTION = "malformed_action"
    MISSING_REWARD = "missing_reward"
    NETWORK = "network"
    OOM_GPU = "oom_gpu"
    OOM_HOST = "oom_host"
    OOM_KV_CACHE = "oom_kv_cache"
    PROCESS_KILLED = "process_killed"
    SANDBOX = "sandbox"
    WALL_CLOCK = "wall_clock"


KIND_CLASS: Final[dict[FailureKind, FailureClass]] = {
    FailureKind.AGENT_EXCEPTION: FailureClass.AGENT,
    FailureKind.ENGINE_CRASH: FailureClass.INFRA,
    FailureKind.GRADER_EXCEPTION: FailureClass.GRADER,
    FailureKind.HARDWARE: FailureClass.INFRA,
    FailureKind.MALFORMED_ACTION: FailureClass.AGENT,
    FailureKind.MISSING_REWARD: FailureClass.GRADER,
    FailureKind.NETWORK: FailureClass.INFRA,
    FailureKind.OOM_GPU: FailureClass.INFRA,
    FailureKind.OOM_HOST: FailureClass.INFRA,
    FailureKind.OOM_KV_CACHE: FailureClass.INFRA,
    FailureKind.PROCESS_KILLED: FailureClass.INFRA,
    FailureKind.SANDBOX: FailureClass.INFRA,
    FailureKind.WALL_CLOCK: FailureClass.TIMEOUT,
}


REWARDED_KINDS: Final[frozenset[UnitKind]] = frozenset({UnitKind.EVAL, UnitKind.TRAJECTORY})


class Outcome(StrEnum):
    """Whether a unit attempt succeeded."""

    FAILED = "failed"
    OK = "ok"


class Action(StrEnum):
    """What the trainer does with one unit attempt."""

    ABORT = "abort"
    MASK = "mask"
    RETRY = "retry"
    SCORE = "score"
    ZERO = "zero"


@dataclass(frozen=True, kw_only=True)
class Failure:
    """One classified non-algorithm failure with sorted, unique evidence lines."""

    kind: FailureKind
    stage: Stage
    retryable: bool
    message: str
    evidence: tuple[str, ...]
    exception_type: str | None = None

    def __post_init__(self) -> None:
        if self.message == "":
            raise SchemaError("failure message must not be empty")
        if list(self.evidence) != sorted(set(self.evidence)):
            raise SchemaError(f"failure evidence must be sorted and unique, got {self.evidence!r}")

    @property
    def failure_class(self) -> FailureClass:
        return KIND_CLASS[self.kind]


@dataclass(frozen=True, kw_only=True)
class ResourceSummary:
    """Peak memory use over one unit, with one GPU entry per sampled device."""

    sample_count: int
    peak_host_rss_bytes: int
    peak_gpu_used_bytes: tuple[int, ...]
    gpu_total_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sample_count < 0 or self.peak_host_rss_bytes < 0:
            raise SchemaError("resource counts must be non-negative")
        if len(self.peak_gpu_used_bytes) != len(self.gpu_total_bytes):
            raise SchemaError(
                f"peak_gpu_used_bytes has {len(self.peak_gpu_used_bytes)} entries but "
                f"gpu_total_bytes has {len(self.gpu_total_bytes)}; both must list every sampled device"
            )


@dataclass(frozen=True, kw_only=True)
class CostLine:
    """Resources held by one unit and their price, when every used resource is priced."""

    cpu_seconds: float
    gpu_seconds: float
    tokens_in: int
    tokens_out: int
    usd: float | None

    def __post_init__(self) -> None:
        quantities = (self.cpu_seconds, self.gpu_seconds, self.tokens_in, self.tokens_out)
        if any(not (math.isfinite(quantity) and quantity >= 0) for quantity in quantities):
            raise SchemaError(
                f"cost quantities must be finite and non-negative, got {quantities!r}"
            )
        if self.usd is not None and (self.usd < 0 or not math.isfinite(self.usd)):
            raise SchemaError(f"cost usd must be a finite non-negative number, got {self.usd!r}")


@dataclass(frozen=True, kw_only=True)
class MismatchSummary:
    """Train-inference mismatch over one batch: KL(rollout||train) per token and the mean importance ratio, each with a sequence-clustered standard error."""

    sequences: int
    tokens: int
    kl: float
    kl_standard_error: float
    ratio_mean: float
    ratio_standard_error: float

    def __post_init__(self) -> None:
        if self.sequences < MIN_MISMATCH_SEQUENCES or self.tokens < self.sequences:
            raise SchemaError(
                f"mismatch needs at least {MIN_MISMATCH_SEQUENCES} sequences and one token per sequence, "
                f"got {self.sequences} sequences and {self.tokens} tokens"
            )
        values = (self.kl, self.kl_standard_error, self.ratio_mean, self.ratio_standard_error)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise SchemaError(
                f"mismatch statistics must be finite and non-negative, got {values!r}"
            )


@dataclass(frozen=True, kw_only=True)
class Decision:
    """What the trainer does with one attempt of a unit."""

    action: Action
    reason: str
    attempt: int

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise SchemaError(f"decision attempt must be at least 1, got {self.attempt}")


@dataclass(frozen=True, kw_only=True)
class ExecResult:
    """How one child process ended: an exit code or a signal, never both."""

    exit_code: int | None
    signal: int | None
    timed_out: bool
    elapsed_ms: int
    stdout_tail: str
    stderr_tail: str

    def __post_init__(self) -> None:
        if (self.exit_code is None) == (self.signal is None):
            raise SchemaError("exactly one of exit_code and signal must be set")
        if self.elapsed_ms < 0:
            raise SchemaError(f"elapsed_ms must be non-negative, got {self.elapsed_ms}")


@dataclass(frozen=True, kw_only=True)
class Record:
    """The single persisted outcome of one attempt of one unit."""

    schema_version: int
    run_id: str
    unit_id: str
    kind: UnitKind
    attempt: int
    source: str | None
    started_at_ms: int
    ended_at_ms: int
    outcome: Outcome
    failure: Failure | None
    reward: float | None
    resources: ResourceSummary
    cost: CostLine
    decision: Decision
    tags: tuple[tuple[str, str], ...]
    mismatch: MismatchSummary | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version}")
        _require_identifier("run_id", self.run_id)
        _require_identifier("unit_id", self.unit_id)
        if self.attempt < 1:
            raise SchemaError(f"attempt must be at least 1, got {self.attempt}")
        if self.decision.attempt != self.attempt:
            raise SchemaError(
                f"decision.attempt must equal attempt, got {self.decision.attempt} and {self.attempt}"
            )
        if self.ended_at_ms < self.started_at_ms:
            raise SchemaError("ended_at_ms must be at or after started_at_ms")
        if (self.outcome is Outcome.FAILED) != (self.failure is not None):
            raise SchemaError("outcome is failed exactly when failure is present")
        _require_consistent_decision(self.kind, self.outcome, self.decision.action, self.reward)
        if self.reward is not None and not math.isfinite(self.reward):
            raise SchemaError(f"reward must be finite, got {self.reward!r}")
        keys = [key for key, _ in self.tags]
        if keys != sorted(set(keys)):
            raise SchemaError(f"tags must be sorted by key and unique, got {self.tags!r}")


def _require_consistent_decision(
    kind: UnitKind, outcome: Outcome, action: Action, reward: float | None
) -> None:
    """Reject a decision that could not come from decide() and reward_for() for this outcome."""
    if (outcome is Outcome.OK) != (action is Action.SCORE):
        raise SchemaError(
            f"decision must be score exactly when outcome is ok, got {action.value} for {outcome.value}"
        )
    if action is Action.ZERO and reward != 0.0:
        raise SchemaError(f"a zero decision requires reward 0.0, got {reward!r}")
    if action in (Action.MASK, Action.RETRY, Action.ABORT) and reward is not None:
        raise SchemaError(f"reward must be absent for mask, retry and abort, got {reward!r}")
    if action is Action.SCORE and kind in REWARDED_KINDS and reward is None:
        raise SchemaError(
            "a scored trajectory or eval must have a reward; a missing reward is a grader failure"
        )


def _require_identifier(name: str, value: str) -> None:
    if IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise SchemaError(f"{name} must match {IDENTIFIER_PATTERN.pattern}, got {value!r}")
