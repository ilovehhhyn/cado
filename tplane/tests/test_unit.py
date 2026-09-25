"""Every unit attempt writes exactly one record then suppresses or re-raises according to the policy, and a trajectory without a reward is a grader failure."""

from __future__ import annotations

import gc
import sys
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from tests.support import MemoryRecordStore
from tplane.classify import classify_exception
from tplane.exec import BASE_ENV
from tplane.policy import DEFAULT_POLICY, Policy
from tplane.resources import ResourceError
from tplane.schema import (
    Action,
    ExecResult,
    FailureClass,
    FailureKind,
    MismatchSummary,
    Outcome,
    Stage,
    UnitKind,
)
from tplane.store import StoreError
from tplane.unit import Run, RunOptions, Unit, UnitAborted, UnitError

PYTHON_ENV = {**BASE_ENV, "PATH": "/usr/bin:/bin"}


class OutOfMemoryError(RuntimeError):
    """Stand-in with torch's class name."""


class SandboxError(RuntimeError):
    """Stand-in for a sandbox SDK error, as in the MiMo incident."""


def stepping_clock(start: int = 1_000, step: int = 10) -> Callable[[], int]:
    current = {"now": start - step}

    def read() -> int:
        current["now"] += step
        return current["now"]

    return read


def make_run(
    *, policy: Policy = DEFAULT_POLICY, gpus_held: int = 1, log: list[str] | None = None
) -> tuple[Run, MemoryRecordStore]:
    store = MemoryRecordStore()
    sink: list[str] = [] if log is None else log
    run = Run(
        run_id="run-1",
        store=store,
        options=RunOptions(
            policy=policy, sample_interval_ms=10, gpus_held=gpus_held, host_limit_bytes=1_000
        ),
        read_host=lambda: 500,
        read_gpu=lambda: (2_048,),
        gpu_total_bytes=(4_096,),
        clock_ms=stepping_clock(),
        monotonic_ms=stepping_clock(),
        log=sink.append,
    )
    return run, store


def test_successful_trajectory_writes_one_scored_record_with_its_reward_and_cost() -> None:
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY, source="swe") as unit:
        with unit.stage(Stage.ROLLOUT):
            unit.tokens(tokens_in=10, tokens_out=5)
        unit.reward(0.75)

    (record,) = store.iterate()
    assert record.outcome is Outcome.OK
    assert record.decision.action is Action.SCORE
    assert record.reward == 0.75
    assert record.source == "swe"
    assert record.cost.tokens_in == 10 and record.cost.tokens_out == 5
    assert record.cost.gpu_seconds > 0
    assert record.resources.peak_gpu_used_bytes == (2_048,)
    assert store.started_without_record() == ()
    assert unit.result.ok is True


def test_trajectory_without_a_reward_is_a_missing_reward_grader_failure() -> None:
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        unit.tokens(tokens_in=1, tokens_out=1)

    (record,) = store.iterate()
    assert record.outcome is Outcome.FAILED
    assert record.failure is not None and record.failure.kind is FailureKind.MISSING_REWARD
    assert record.failure.stage is Stage.GRADE
    assert record.decision.action is Action.MASK
    assert record.reward is None


def test_train_step_without_a_reward_is_fine() -> None:
    run, store = make_run()

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP):
        pass

    assert store.iterate()[0].outcome is Outcome.OK


def test_sandbox_error_in_a_rollout_is_recorded_as_infra_masked_and_never_scored() -> None:
    # Invariant: the MiMo incident cannot happen; a sandbox failure never reaches the reward.
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY, source="swe") as unit:
        with unit.stage(Stage.ROLLOUT):
            raise SandboxError("sandbox connection refused: missing dependency")

    (record,) = store.iterate()
    assert record.failure is not None
    assert (record.failure.failure_class, record.failure.kind) == (
        FailureClass.INFRA,
        FailureKind.SANDBOX,
    )
    assert record.decision.action is Action.MASK
    assert record.reward is None


def test_infra_exception_is_masked_suppressed_and_classified_with_the_innermost_stage() -> None:
    run, store = make_run()
    reached_after = False

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        with unit.stage(Stage.ROLLOUT):
            with unit.stage(Stage.TOOL):
                raise OutOfMemoryError("CUDA out of memory")
    reached_after = True

    (record,) = store.iterate()
    assert reached_after is True
    assert record.failure is not None
    assert record.failure.kind is FailureKind.OOM_GPU
    assert record.failure.stage is Stage.TOOL
    assert record.decision.action is Action.MASK
    assert record.reward is None


def test_an_error_handled_inside_a_stage_does_not_label_a_later_error() -> None:
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        try:
            with unit.stage(Stage.TOOL):
                raise ValueError("handled by the agent")
        except ValueError:
            pass
        raise KeyError("later bug")

    (record,) = store.iterate()
    assert record.failure is not None
    assert record.failure.stage is Stage.SETUP
    assert record.failure.exception_type == "KeyError"


def test_agent_exception_is_zero_scored_and_suppressed() -> None:
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY):
        raise ValueError("bad action json")

    (record,) = store.iterate()
    assert record.failure is not None and record.failure.kind is FailureKind.AGENT_EXCEPTION
    assert record.decision.action is Action.ZERO
    assert record.reward == 0.0


def test_abort_policy_writes_the_record_then_re_raises() -> None:
    run, store = make_run(policy=Policy(on_agent=Action.ABORT))

    with pytest.raises(ValueError, match="bad action json"):
        with run.unit("t-1", kind=UnitKind.TRAJECTORY):
            raise ValueError("bad action json")

    (record,) = store.iterate()
    assert record.decision.action is Action.ABORT


@pytest.mark.parametrize("interrupt", (KeyboardInterrupt, SystemExit))
def test_interrupts_propagate_without_a_record_and_keep_the_start_marker(
    interrupt: type[BaseException],
) -> None:
    run, store = make_run()

    with pytest.raises(interrupt):
        with run.unit("t-1", kind=UnitKind.TRAJECTORY):
            raise interrupt()

    assert store.iterate() == ()
    assert [marker.unit_id for marker in store.started_without_record()] == ["t-1"]


def test_unit_fail_ends_the_unit_with_the_given_failure() -> None:
    run, store = make_run()
    failure = classify_exception(ConnectionRefusedError(111, "refused"), stage=Stage.GRADE)

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        unit.fail(failure)

    (record,) = store.iterate()
    assert record.failure == failure


def test_attempt_retries_a_retryable_failure_then_masks_with_one_record_per_attempt() -> None:
    run, store = make_run(policy=Policy(max_attempts=3))
    calls = {"count": 0}

    def body(unit: Unit) -> None:
        calls["count"] += 1
        raise ConnectionRefusedError(111, "refused")

    result = run.attempt("t-1", kind=UnitKind.TRAJECTORY, body=body)

    assert calls["count"] == 3
    assert [record.decision.action for record in store.iterate()] == [
        Action.RETRY,
        Action.RETRY,
        Action.MASK,
    ]
    assert result.record.attempt == 3


def test_attempt_returns_the_first_successful_result() -> None:
    run, store = make_run(policy=Policy(max_attempts=3))
    calls = {"count": 0}

    def body(unit: Unit) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionRefusedError(111, "refused")
        unit.reward(1.0)

    result = run.attempt("t-1", kind=UnitKind.TRAJECTORY, body=body)

    assert result.ok is True
    assert result.record.attempt == 2
    assert len(store.iterate()) == 2


def test_unit_rejects_an_attempt_outside_the_policy_before_any_effect() -> None:
    run, store = make_run(policy=Policy(max_attempts=2))

    for attempt in (0, 3):
        with pytest.raises(UnitError, match=f"attempt must be between 1 and 2, got {attempt}"):
            run.unit("t-1", kind=UnitKind.TRAJECTORY, attempt=attempt)

    assert store.markers == {} and store.records == []


def test_exec_that_times_out_ends_the_unit_with_a_wall_clock_failure() -> None:
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        unit.exec(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_s=0.3,
            env=PYTHON_ENV,
            stage=Stage.TOOL,
        )
        unit.reward(1.0)

    (record,) = store.iterate()
    assert record.failure is not None and record.failure.kind is FailureKind.WALL_CLOCK
    assert record.failure.stage is Stage.TOOL


def test_exec_with_a_non_zero_exit_returns_the_result_to_the_caller() -> None:
    run, store = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        result = unit.exec(
            [sys.executable, "-c", "raise SystemExit(2)"], timeout_s=5, env=PYTHON_ENV
        )
        unit.reward(0.0 if result.exit_code != 0 else 1.0)

    assert result.exit_code == 2
    assert store.iterate()[0].reward == 0.0


def test_result_before_exit_is_an_error() -> None:
    run, _ = make_run()
    unit = run.unit("t-1", kind=UnitKind.TRAJECTORY)

    with pytest.raises(UnitError, match="unit t-1 has not finished"):
        _ = unit.result


def test_unit_rejects_non_finite_reward_and_negative_tokens() -> None:
    run, _ = make_run()

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        with pytest.raises(UnitError, match="reward must be finite, got nan"):
            unit.reward(float("nan"))
        with pytest.raises(UnitError, match="token counts must be non-negative, got -1 and 0"):
            unit.tokens(tokens_in=-1, tokens_out=0)
        unit.reward(1.0)


def test_run_rejects_gpus_held_without_a_gpu_reader() -> None:
    with pytest.raises(UnitError, match="gpus_held is 2 but gpu_total_bytes lists 0 devices"):
        Run(
            run_id="run-1",
            store=MemoryRecordStore(),
            options=RunOptions(gpus_held=2),
            read_host=lambda: 0,
            read_gpu=lambda: (),
            gpu_total_bytes=(),
        )


def test_run_rejects_a_run_id_with_a_slash() -> None:
    with pytest.raises(UnitError, match="run_id must match"):
        Run(
            run_id="a/b",
            store=MemoryRecordStore(),
            read_host=lambda: 0,
            read_gpu=lambda: (),
            gpu_total_bytes=(),
        )


def test_run_options_reject_values_that_cannot_be_honoured() -> None:
    with pytest.raises(UnitError, match="sample_interval_ms must be at least 10, got 1"):
        RunOptions(sample_interval_ms=1)
    with pytest.raises(UnitError, match="host_limit_bytes must be positive or None, got 0"):
        RunOptions(host_limit_bytes=0)


def test_log_receives_one_line_per_record() -> None:
    lines: list[str] = []
    run, _ = make_run(log=lines)

    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        unit.reward(1.0)

    assert lines == ["tplane t-1 attempt 1 ok score -"]


class Payload:
    """Stands in for the GPU tensors a failing step holds."""


def test_a_failed_unit_releases_the_failing_frame_without_waiting_for_the_garbage_collector() -> (
    None
):
    # Invariant: a failed step's locals are freed when its unit exits; otherwise the next step runs out of memory too.
    # Witness: Della job 14383608, where steps 2 and 3 ran out of memory at their first allocation.
    run, _ = make_run()
    references: list[weakref.ref[Payload]] = []

    def body(unit: Unit) -> None:
        payload = Payload()
        references.append(weakref.ref(payload))
        with unit.stage(Stage.TRAIN):
            raise OutOfMemoryError("CUDA out of memory")

    gc.disable()
    try:
        run.attempt("s-1", kind=UnitKind.TRAIN_STEP, body=body)
        alive = references[0]() is not None
    finally:
        gc.enable()

    assert len(references) == 1
    assert alive is False


def test_a_train_step_persists_its_mismatch_summary_once() -> None:
    run, store = make_run()
    summary = MismatchSummary(
        sequences=4,
        tokens=40,
        kl=1e-3,
        kl_standard_error=1e-4,
        ratio_mean=1.0,
        ratio_standard_error=1e-3,
    )

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP) as unit:
        unit.mismatch(summary)
        with pytest.raises(
            UnitError, match="unit s-1 already has a mismatch summary; record one per train step"
        ):
            unit.mismatch(summary)

    assert store.iterate()[0].mismatch == summary


def test_run_probes_an_unreadable_host_source_before_any_work() -> None:
    def unreadable() -> int:
        raise ResourceError("/sys/fs/cgroup/job_1/memory.current: No such file or directory")

    with pytest.raises(
        UnitError,
        match="read_host failed when the run started: /sys/fs/cgroup/job_1/memory.current",
    ):
        Run(
            run_id="run-1",
            store=MemoryRecordStore(),
            read_host=unreadable,
            read_gpu=lambda: (),
            gpu_total_bytes=(),
        )


def test_run_rejects_a_gpu_reader_whose_device_count_disagrees_with_the_totals() -> None:
    with pytest.raises(
        UnitError, match="read_gpu returned 2 entries but gpu_total_bytes lists 1 device"
    ):
        Run(
            run_id="run-1",
            store=MemoryRecordStore(),
            read_host=lambda: 0,
            read_gpu=lambda: (1, 2),
            gpu_total_bytes=(10,),
        )


def test_a_gpu_reader_that_changes_device_count_mid_unit_still_leaves_one_record() -> None:
    # Invariant: a monitoring fault is recorded as a sampler error, never raised out of the unit.
    counts = iter([(1,)] + [(1, 2)] * 1_000)
    store = MemoryRecordStore()
    run = Run(
        run_id="run-1",
        store=store,
        options=RunOptions(sample_interval_ms=10),
        read_host=lambda: 1,
        read_gpu=lambda: next(counts),
        gpu_total_bytes=(10,),
    )

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP):
        pass

    (record,) = store.iterate()
    assert record.outcome is Outcome.OK
    assert dict(record.tags)["sampler_errors"] == "1"


def test_a_wall_clock_that_steps_back_still_leaves_one_record_with_a_monotonic_duration() -> None:
    # Witness: time.time stepping back 1 ms made CostLine reject a negative duration inside __exit__.
    wall = iter([5_000, 5_000, 4_999, 4_999, 4_999, 4_999])
    monotonic = iter([100, 350, 350, 350])
    store = MemoryRecordStore()
    run = Run(
        run_id="run-1",
        store=store,
        options=RunOptions(sample_interval_ms=10_000, cpus_held=2),
        read_host=lambda: 1,
        read_gpu=lambda: (),
        gpu_total_bytes=(),
        clock_ms=lambda: next(wall),
        monotonic_ms=lambda: next(monotonic),
    )

    with run.unit("s-1", kind=UnitKind.TRAIN_STEP):
        pass

    (record,) = store.iterate()
    assert (record.started_at_ms, record.ended_at_ms) == (5_000, 5_250)
    assert record.cost.cpu_seconds == 0.5


def test_misuse_of_the_tplane_api_inside_a_unit_propagates_instead_of_being_zeroed_as_the_agent() -> (
    None
):
    # Witness: unit.reward(nan) was recorded as agent_exception with decision zero and reward 0.0.
    run, store = make_run()

    with pytest.raises(UnitError, match="reward must be finite"):
        with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
            unit.reward(float("nan"))

    assert store.iterate() == ()
    assert [marker.unit_id for marker in store.started_without_record()] == ["t-1"]


def test_an_abort_decision_without_an_exception_raises_unit_aborted_after_writing_the_record() -> (
    None
):
    # Witness: a missing reward under on_grader=abort fell through the with block and the run went on.
    run, store = make_run(policy=Policy(on_grader=Action.ABORT))

    with pytest.raises(
        UnitAborted, match="unit t-1 attempt 1 was aborted: grader failure missing_reward"
    ):
        run.attempt("t-1", kind=UnitKind.TRAJECTORY, body=lambda unit: None)

    (record,) = store.iterate()
    assert record.decision.action is Action.ABORT
    assert store.started_without_record() == ()


def test_a_unit_that_already_finished_is_refused_before_its_body_runs() -> None:
    # Witness: re-running a finished unit executed its whole body, then failed to write the record.
    run, _ = make_run()
    calls = {"count": 0}

    def body(unit: Unit) -> None:
        calls["count"] += 1
        unit.reward(1.0)

    run.attempt("t-1", kind=UnitKind.TRAJECTORY, body=body)
    with pytest.raises(StoreError, match="unit t-1 attempt 1 already has a record"):
        run.attempt("t-1", kind=UnitKind.TRAJECTORY, body=body)

    assert calls["count"] == 1


class RefusingExec:
    """ExecFunction fake that records calls; the tests assert it was never reached."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> ExecResult:
        self.calls.append(list(argv))
        raise AssertionError(f"exec must not run, got {argv!r}")


def test_exec_outside_the_with_block_is_refused_before_the_child_starts() -> None:
    fake = RefusingExec()
    run = Run(
        run_id="run-1",
        store=MemoryRecordStore(),
        read_host=lambda: 1,
        read_gpu=lambda: (),
        gpu_total_bytes=(),
        exec_=fake,
    )
    unit = run.unit("t-1", kind=UnitKind.TRAJECTORY)

    with pytest.raises(UnitError, match="unit t-1 is not open; call it inside its with block"):
        unit.exec(["rm", "-rf", "x"], timeout_s=1, env={})

    assert fake.calls == []


def test_calls_after_the_with_block_are_refused_not_ignored() -> None:
    run, store = make_run()
    with run.unit("t-1", kind=UnitKind.TRAJECTORY) as unit:
        unit.reward(1.0)

    for call in (
        lambda: unit.reward(0.0),
        lambda: unit.tag("k", "v"),
        lambda: unit.tokens(tokens_in=1, tokens_out=1),
    ):
        with pytest.raises(UnitError, match="unit t-1 is not open"):
            call()

    assert store.iterate()[0].reward == 1.0


def test_a_unit_is_entered_at_most_once() -> None:
    run, _ = make_run()
    unit = run.unit("t-1", kind=UnitKind.TRAJECTORY)
    with unit:
        unit.reward(1.0)

    with pytest.raises(
        UnitError,
        match="unit t-1 attempt 1 was already entered; create a new unit for each attempt",
    ):
        with unit:
            pass
