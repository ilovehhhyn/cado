"""Own the lifecycle of one run and its units: sample, classify, decide, persist exactly one record per attempt, then suppress or re-raise."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Final, NoReturn, TypeVar

from tplane.classify import classify_exception, classify_exec
from tplane.cost import NO_PRICES, PriceError, PriceTable
from tplane.exec import ExecFunction, exec_process, monotonic_ms
from tplane.guard import GuardError, headroom_warning
from tplane.mismatch import MismatchError
from tplane.policy import DEFAULT_POLICY, Policy, PolicyError, decide
from tplane.record import MILLISECONDS_PER_SECOND, build_record, render_log_line
from tplane.resources import Sampler, summarise_samples
from tplane.schema import (
    IDENTIFIER_PATTERN,
    REWARDED_KINDS,
    Action,
    ExecResult,
    Failure,
    FailureKind,
    MismatchSummary,
    Outcome,
    Record,
    SchemaError,
    Stage,
    UnitKind,
)
from tplane.store import RecordStore

DEFAULT_SAMPLE_INTERVAL_MS: Final[int] = 250
MIN_SAMPLE_INTERVAL_MS: Final[int] = 10
HEADROOM_UNKNOWN: Final[str] = "unknown; no gpu sample has been taken"

Value = TypeVar("Value")


class _UnitState(StrEnum):
    """Where one unit is in its single pass through the with block."""

    CLOSED = "closed"
    NEW = "new"
    OPEN = "open"


class UnitError(RuntimeError):
    """A run and its units are used in the documented order with valid settings."""


class UnitFailed(RuntimeError):
    """Raised inside a unit to end it with an already-classified failure."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class UnitAborted(RuntimeError):
    """Raised after a unit wrote an abort record while no exception was in flight, so the run stops."""

    def __init__(self, record: Record) -> None:
        super().__init__(
            f"unit {record.unit_id} attempt {record.attempt} was aborted: {record.decision.reason}"
        )
        self.record = record


@dataclass(frozen=True, kw_only=True)
class RunOptions:
    """Settings shared by every unit of a run."""

    policy: Policy = DEFAULT_POLICY
    prices: PriceTable = NO_PRICES
    sample_interval_ms: int = DEFAULT_SAMPLE_INTERVAL_MS
    host_limit_bytes: int | None = None
    cpus_held: int = 1
    gpus_held: int = 0

    def __post_init__(self) -> None:
        if self.sample_interval_ms < MIN_SAMPLE_INTERVAL_MS:
            raise UnitError(
                f"sample_interval_ms must be at least {MIN_SAMPLE_INTERVAL_MS}, got {self.sample_interval_ms}"
            )
        if self.cpus_held < 1 or self.gpus_held < 0:
            raise UnitError(
                f"cpus_held must be at least 1 and gpus_held at least 0, got {self.cpus_held} and {self.gpus_held}"
            )
        if self.host_limit_bytes is not None and self.host_limit_bytes < 1:
            raise UnitError(
                f"host_limit_bytes must be positive or None, got {self.host_limit_bytes}"
            )


DEFAULT_RUN_OPTIONS: Final[RunOptions] = RunOptions()


@dataclass(frozen=True, kw_only=True)
class UnitResult:
    """The persisted record of one finished attempt."""

    record: Record

    @property
    def ok(self) -> bool:
        return self.record.outcome is Outcome.OK


def wall_clock_ms() -> int:
    """Return Unix time in milliseconds."""
    return int(time.time() * MILLISECONDS_PER_SECOND)


def _discard(line: str) -> None:
    del line


def _caller_errors() -> tuple[type[Exception], ...]:
    """Return the errors that mean tplane was used wrongly; they are the caller's bug, never the agent's."""
    return (GuardError, MismatchError, PolicyError, PriceError, SchemaError, UnitError)


def _probe(name: str, read: Callable[[], Value]) -> Value:
    """Read a configured source once, so an unreadable source fails when the run starts, not mid-rollout."""
    try:
        return read()
    except Exception as error:
        raise UnitError(
            f"{name} failed when the run started: {error}; pass a readable source"
        ) from error


class Run:
    """One training run: the store, the policy and the effect providers every unit shares."""

    def __init__(
        self,
        *,
        run_id: str,
        store: RecordStore,
        options: RunOptions = DEFAULT_RUN_OPTIONS,
        read_host: Callable[[], int],
        read_gpu: Callable[[], tuple[int, ...]],
        gpu_total_bytes: tuple[int, ...],
        clock_ms: Callable[[], int] = wall_clock_ms,
        monotonic_ms: Callable[[], int] = monotonic_ms,
        exec_: ExecFunction = exec_process,
        log: Callable[[str], None] = _discard,
    ) -> None:
        if IDENTIFIER_PATTERN.fullmatch(run_id) is None:
            raise UnitError(f"run_id must match {IDENTIFIER_PATTERN.pattern}, got {run_id!r}")
        if options.gpus_held > len(gpu_total_bytes):
            raise UnitError(
                f"gpus_held is {options.gpus_held} but gpu_total_bytes lists {len(gpu_total_bytes)} devices; "
                "pass read_gpu and gpu_total_bytes from create_nvml_gpu_reader() or lower gpus_held"
            )
        _probe("read_host", read_host)
        probed = _probe("read_gpu", read_gpu)
        if len(probed) != len(gpu_total_bytes):
            noun = "device" if len(gpu_total_bytes) == 1 else "devices"
            raise UnitError(
                f"read_gpu returned {len(probed)} entries but gpu_total_bytes lists {len(gpu_total_bytes)} {noun}; "
                "take both from the same reader"
            )
        self.run_id = run_id
        self.options = options
        self._store = store
        self._read_host = read_host
        self._read_gpu = read_gpu
        self._gpu_total_bytes = gpu_total_bytes
        self._clock_ms = clock_ms
        self._monotonic_ms = monotonic_ms
        self._exec = exec_
        self._log = log

    def unit(
        self, unit_id: str, *, kind: UnitKind, source: str | None = None, attempt: int = 1
    ) -> Unit:
        """Create one attempt of a unit; enter it with `with`."""
        return Unit(self, unit_id=unit_id, kind=kind, source=source, attempt=attempt)

    def attempt(
        self,
        unit_id: str,
        *,
        kind: UnitKind,
        body: Callable[[Unit], None],
        source: str | None = None,
    ) -> UnitResult:
        """Run body once per attempt until the decision is not retry, returning the final attempt's result."""
        for attempt_number in range(1, self.options.policy.max_attempts + 1):
            unit = self.unit(unit_id, kind=kind, source=source, attempt=attempt_number)
            with unit:
                body(unit)
            if unit.result.record.decision.action is not Action.RETRY:
                return unit.result
        raise UnitError(
            "decide returned retry on the final attempt; the policy contract forbids this"
        )


class Unit:
    """One attempt of one unit of work; a context manager that always leaves exactly one record or re-raises."""

    def __init__(
        self, run: Run, *, unit_id: str, kind: UnitKind, source: str | None, attempt: int
    ) -> None:
        if IDENTIFIER_PATTERN.fullmatch(unit_id) is None:
            raise UnitError(f"unit_id must match {IDENTIFIER_PATTERN.pattern}, got {unit_id!r}")
        max_attempts = run.options.policy.max_attempts
        if attempt < 1 or attempt > max_attempts:
            raise UnitError(f"attempt must be between 1 and {max_attempts}, got {attempt}")
        self._run = run
        self._unit_id = unit_id
        self._kind = kind
        self._source = source
        self._attempt = attempt
        self._stage = Stage.SETUP
        self._failed_in: tuple[BaseException, Stage] | None = None
        self._reward: float | None = None
        self._mismatch: MismatchSummary | None = None
        self._tokens_in = 0
        self._tokens_out = 0
        self._tags: dict[str, str] = {}
        self._sampler: Sampler | None = None
        self._started_at_ms = 0
        self._started_monotonic_ms = 0
        self._result: UnitResult | None = None
        self._state = _UnitState.NEW

    @property
    def unit_id(self) -> str:
        return self._unit_id

    @property
    def result(self) -> UnitResult:
        """Return the finished attempt's result; only valid after the with block."""
        if self._result is None:
            raise UnitError(
                f"unit {self._unit_id} has not finished; read result after the with block"
            )
        return self._result

    def __enter__(self) -> Unit:
        if self._state is not _UnitState.NEW:
            raise UnitError(
                f"unit {self._unit_id} attempt {self._attempt} was already entered; create a new unit for each attempt"
            )
        self._started_at_ms = self._run._clock_ms()
        self._started_monotonic_ms = self._run._monotonic_ms()
        self._run._store.mark_started(self._unit_id, self._attempt, self._started_at_ms)
        self._sampler = Sampler(
            read_host=self._run._read_host,
            read_gpu=self._run._read_gpu,
            interval_ms=self._run.options.sample_interval_ms,
            clock_ms=self._run._clock_ms,
            gpu_devices=len(self._run._gpu_total_bytes),
        )
        self._sampler.start()
        self._state = _UnitState.OPEN
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._state = _UnitState.CLOSED
        samples = self._require_sampler().stop()
        # KeyboardInterrupt, SystemExit and misuse of tplane itself leave no record; the start marker reports the unit.
        if exc is not None and (
            not isinstance(exc, Exception) or isinstance(exc, _caller_errors())
        ):
            self._failed_in = None
            return False
        failure = self._failure_from(exc)
        # The stored exception's traceback holds the failing frame, and the frame holds this unit.
        # Dropping the reference here frees the failed step's memory (its GPU tensors) now, not at the next GC.
        self._failed_in = None
        decision = decide(failure, policy=self._run.options.policy, attempt=self._attempt)
        record = build_record(
            run_id=self._run.run_id,
            unit_id=self._unit_id,
            kind=self._kind,
            attempt=self._attempt,
            source=self._source,
            started_at_ms=self._started_at_ms,
            # Durations come from the monotonic clock, so a wall clock stepping back cannot make end precede start.
            ended_at_ms=self._started_at_ms
            + self._run._monotonic_ms()
            - self._started_monotonic_ms,
            failure=failure,
            reward=self._reward,
            resources=summarise_samples(samples, gpu_total_bytes=self._run._gpu_total_bytes),
            decision=decision,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            prices=self._run.options.prices,
            cpus_held=self._run.options.cpus_held,
            gpus_held=self._run.options.gpus_held,
            tags=self._tags_with_sampler_errors(),
            mismatch=self._mismatch,
        )
        self._run._store.append(record)
        self._run._store.clear_started(self._unit_id, self._attempt)
        self._result = UnitResult(record=record)
        self._run._log(render_log_line(record))
        if decision.action is Action.ABORT and exc is None:
            raise UnitAborted(record)
        return decision.action is not Action.ABORT

    @contextmanager
    def stage(self, stage: Stage) -> Iterator[None]:
        """Run the block as this stage; a failure inside is attributed to the innermost stage."""
        self._require_open()
        previous = self._stage
        self._stage = stage
        try:
            yield
        except BaseException as error:
            if self._failed_in is None or self._failed_in[0] is not error:
                self._failed_in = (error, stage)
            raise
        finally:
            self._stage = previous

    def tokens(self, *, tokens_in: int, tokens_out: int) -> None:
        """Add consumed and generated tokens to this unit's cost line."""
        self._require_open()
        if tokens_in < 0 or tokens_out < 0:
            raise UnitError(f"token counts must be non-negative, got {tokens_in} and {tokens_out}")
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out

    def reward(self, value: float) -> None:
        """Set the unit's reward; a trajectory or eval must set one before exit."""
        self._require_open()
        if not math.isfinite(value):
            raise UnitError(f"reward must be finite, got {value!r}")
        self._reward = value

    def mismatch(self, summary: MismatchSummary) -> None:
        """Attach the batch's train-inference mismatch summary; a unit carries at most one."""
        self._require_open()
        if self._mismatch is not None:
            raise UnitError(
                f"unit {self._unit_id} already has a mismatch summary; record one per train step"
            )
        self._mismatch = summary

    def tag(self, key: str, value: str) -> None:
        """Attach one string tag to the record; a later value for the same key replaces the earlier one."""
        self._require_open()
        self._tags[key] = value

    def fail(self, failure: Failure) -> NoReturn:
        """End the unit now with an already-classified failure."""
        self._require_open()
        raise UnitFailed(failure)

    def exec(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
        stage: Stage = Stage.TOOL,
    ) -> ExecResult:
        """Run a child under this stage; a timeout, signal or OOM ends the unit, a non-zero exit is returned."""
        self._require_open()
        with self.stage(stage):
            result = self._run._exec(argv, timeout_s=timeout_s, env=env, cwd=cwd)
            partial = summarise_samples(
                self._require_sampler().snapshot(), gpu_total_bytes=self._run._gpu_total_bytes
            )
            failure = classify_exec(
                result,
                resources=partial,
                host_limit_bytes=self._run.options.host_limit_bytes,
                stage=stage,
            )
            if failure is not None:
                raise UnitFailed(failure)
        return result

    def check_headroom(self, *, longest_tokens: int, device_index: int = 0) -> str | None:
        """Log, tag and return a headroom warning from the latest GPU sample; None when headroom is fine or unknown."""
        self._require_open()
        totals = self._run._gpu_total_bytes
        if device_index < 0 or device_index >= len(totals):
            noun = "device is" if len(totals) == 1 else "devices are"
            raise GuardError(
                f"device_index {device_index} is not sampled; {len(totals)} {noun} configured"
            )
        samples = self._require_sampler().snapshot()
        # A monitoring gap must never end the unit, so an absent sample is reported, not raised.
        if len(samples) == 0:
            self._note_headroom(HEADROOM_UNKNOWN)
            return None
        warning = headroom_warning(
            gpu_used_bytes=samples[-1].gpu_used_bytes[device_index],
            gpu_total_bytes=totals[device_index],
            longest_tokens=longest_tokens,
        )
        if warning is not None:
            self._note_headroom(warning)
        return warning

    def _note_headroom(self, text: str) -> None:
        self._tags["headroom_warning"] = text
        self._run._log(f"tplane {self._unit_id} headroom: {text}")

    def _require_open(self) -> None:
        if self._state is not _UnitState.OPEN:
            raise UnitError(f"unit {self._unit_id} is not open; call it inside its with block")

    def _require_sampler(self) -> Sampler:
        if self._sampler is None:
            raise UnitError(f"unit {self._unit_id} must be entered with `with` before use")
        return self._sampler

    def _failure_from(self, exc: BaseException | None) -> Failure | None:
        if isinstance(exc, UnitFailed):
            return exc.failure
        if exc is not None:
            failed_here = self._failed_in is not None and self._failed_in[0] is exc
            stage = (
                self._failed_in[1] if self._failed_in is not None and failed_here else self._stage
            )
            return classify_exception(exc, stage=stage)
        if self._kind in REWARDED_KINDS and self._reward is None:
            return Failure(
                kind=FailureKind.MISSING_REWARD,
                stage=Stage.GRADE,
                retryable=False,
                message="missing_reward: unit ended without a reward; call unit.reward(value) or unit.fail(failure)",
                evidence=("reward=None",),
            )
        return None

    def _tags_with_sampler_errors(self) -> tuple[tuple[str, str], ...]:
        tags = dict(self._tags)
        errors = self._require_sampler().errors
        if len(errors) != 0:
            tags["sampler_errors"] = str(len(errors))
        return tuple(sorted(tags.items()))
