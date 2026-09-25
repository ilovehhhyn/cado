"""Known infrastructure signatures classify as infra or timeout, everything else classifies as agent, and a plain non-zero exit is not a failure."""

from __future__ import annotations

import pytest

from tests.support import make_resources
from tplane.classify import (
    MAX_CAUSE_DEPTH,
    RETRYABLE_KINDS,
    classify_error_name,
    classify_exception,
    classify_exec,
)
from tplane.schema import ExecResult, FailureClass, FailureKind, Stage


class OutOfMemoryError(RuntimeError):
    """Stand-in with torch's class name; the classifier must match by name."""


class SandboxError(RuntimeError):
    """Stand-in for a sandbox SDK error."""


class SandboxUnreachableError(ConnectionError):
    """A third-party subclass of a built-in network error."""


EXCEPTION_CASES: tuple[tuple[BaseException, FailureKind], ...] = (
    (OutOfMemoryError("CUDA out of memory. Tried to allocate 2.50 GiB"), FailureKind.OOM_GPU),
    (RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"), FailureKind.OOM_GPU),
    (MemoryError(), FailureKind.OOM_HOST),
    (RuntimeError("Not enough KV cache blocks for request"), FailureKind.OOM_KV_CACHE),
    (
        RuntimeError("NCCL timeout: watchdog caught collective operation timeout"),
        FailureKind.NETWORK,
    ),
    (ConnectionRefusedError(111, "Connection refused"), FailureKind.NETWORK),
    (ConnectionResetError(104, "Connection reset by peer"), FailureKind.NETWORK),
    (SandboxUnreachableError("sandbox host unreachable"), FailureKind.NETWORK),
    (TimeoutError("request timed out"), FailureKind.WALL_CLOCK),
    (
        RuntimeError("NVRM: Xid (PCI:0000:3b:00): 79, GPU has fallen off the bus"),
        FailureKind.HARDWARE,
    ),
    (SandboxError("sandbox terminated"), FailureKind.SANDBOX),
    (ValueError("could not parse action JSON"), FailureKind.AGENT_EXCEPTION),
    (KeyError("reward"), FailureKind.AGENT_EXCEPTION),
)


@pytest.mark.parametrize(
    ("error", "kind"),
    EXCEPTION_CASES,
    ids=[f"{type(case[0]).__name__}:{str(case[0])[:20]}" for case in EXCEPTION_CASES],
)
def test_exception_signature_table_classifies_each_case(
    error: BaseException, kind: FailureKind
) -> None:
    failure = classify_exception(error, stage=Stage.ROLLOUT)

    assert failure.kind is kind
    assert failure.stage is Stage.ROLLOUT
    assert failure.retryable == (kind in RETRYABLE_KINDS)
    assert f"exception_type={type(error).__name__}" in failure.evidence


def test_unknown_exception_defaults_to_agent_not_infra() -> None:
    failure = classify_exception(RuntimeError("something new"), stage=Stage.TOOL)

    assert failure.failure_class is FailureClass.AGENT
    assert failure.kind is FailureKind.AGENT_EXCEPTION
    assert failure.retryable is False


def test_signature_is_found_through_the_cause_chain() -> None:
    inner = OutOfMemoryError("CUDA out of memory")
    outer = RuntimeError("agent loop failed")
    outer.__cause__ = inner

    failure = classify_exception(outer, stage=Stage.TRAIN)

    assert failure.kind is FailureKind.OOM_GPU
    assert failure.exception_type == "OutOfMemoryError"


def test_cause_chain_walk_is_bounded_and_cycle_safe() -> None:
    first = RuntimeError("0")
    current = first
    for depth in range(1, MAX_CAUSE_DEPTH + 3):
        following = RuntimeError(str(depth))
        current.__cause__ = following
        current = following
    current.__cause__ = first
    current.__context__ = OutOfMemoryError("CUDA out of memory")

    failure = classify_exception(first, stage=Stage.TRAIN)

    assert failure.kind is FailureKind.AGENT_EXCEPTION


def test_evidence_message_line_is_bounded_and_first_line_only() -> None:
    failure = classify_exception(RuntimeError("x" * 1_000 + "\nsecond line"), stage=Stage.ROLLOUT)

    message_lines = [line for line in failure.evidence if line.startswith("message=")]
    assert len(message_lines) == 1
    assert len(message_lines[0]) <= len("message=") + 200
    assert "second line" not in message_lines[0]


def exec_result(
    *,
    exit_code: int | None = 0,
    signal: int | None = None,
    timed_out: bool = False,
    stderr: str = "",
) -> ExecResult:
    return ExecResult(
        exit_code=exit_code,
        signal=signal,
        timed_out=timed_out,
        elapsed_ms=1_500,
        stdout_tail="",
        stderr_tail=stderr,
    )


def test_exec_zero_and_non_zero_exit_are_not_failures() -> None:
    resources = make_resources()

    for code in (0, 1, 2, 137):
        result = exec_result(exit_code=code)
        assert (
            classify_exec(result, resources=resources, host_limit_bytes=None, stage=Stage.TOOL)
            is None
        )


def test_exec_timeout_classifies_as_wall_clock_with_elapsed_evidence() -> None:
    result = exec_result(exit_code=None, signal=9, timed_out=True)

    failure = classify_exec(
        result, resources=make_resources(), host_limit_bytes=None, stage=Stage.TOOL
    )

    assert failure is not None
    assert failure.kind is FailureKind.WALL_CLOCK
    assert "elapsed_ms=1500" in failure.evidence


def test_exec_sigkill_near_host_limit_classifies_as_oom_host() -> None:
    resources = make_resources()
    near_limit = int(resources.peak_host_rss_bytes / 0.9)

    failure = classify_exec(
        exec_result(exit_code=None, signal=9),
        resources=resources,
        host_limit_bytes=near_limit,
        stage=Stage.ROLLOUT,
    )

    assert failure is not None
    assert failure.kind is FailureKind.OOM_HOST
    assert f"peak_host_rss_bytes={resources.peak_host_rss_bytes}" in failure.evidence


def test_exec_sigkill_far_from_host_limit_classifies_as_process_killed() -> None:
    resources = make_resources()
    far_limit = resources.peak_host_rss_bytes * 100

    failure = classify_exec(
        exec_result(exit_code=None, signal=9),
        resources=resources,
        host_limit_bytes=far_limit,
        stage=Stage.ROLLOUT,
    )

    assert failure is not None
    assert failure.kind is FailureKind.PROCESS_KILLED
    assert failure.retryable is True


def test_exec_sigkill_with_oom_killer_text_classifies_as_oom_host_without_a_limit() -> None:
    result = exec_result(exit_code=None, signal=9, stderr="Killed")

    failure = classify_exec(
        result, resources=make_resources(), host_limit_bytes=None, stage=Stage.ROLLOUT
    )

    assert failure is not None
    assert failure.kind is FailureKind.OOM_HOST


def test_exec_cuda_oom_text_classifies_as_oom_gpu_even_with_exit_code() -> None:
    result = exec_result(exit_code=1, stderr="torch.OutOfMemoryError: CUDA out of memory")

    failure = classify_exec(
        result, resources=make_resources(), host_limit_bytes=None, stage=Stage.ROLLOUT
    )

    assert failure is not None
    assert failure.kind is FailureKind.OOM_GPU
    assert "peak_gpu_used_bytes=[1024]" in failure.evidence


def stand_in(name: str, *bases: type[BaseException], module: str = "sdk") -> type[BaseException]:
    """Build an exception class with a real library's name, bases and module."""
    return type(name, bases or (Exception,), {"__module__": module})


E2B_SANDBOX = stand_in("SandboxException", module="e2b.exceptions")
DOCKER = stand_in("DockerException", module="docker.errors")
AIOHTTP_CONNECTION = stand_in("ClientConnectionError", module="aiohttp")
HTTPX_TRANSPORT = stand_in("TransportError", module="httpx")
REQUESTS_TIMEOUT = stand_in("Timeout", OSError, module="requests.exceptions")
RAY_ACTOR = stand_in("RayActorError", module="ray.exceptions")
PROVIDER = stand_in("ProviderError", module="verifiers.v1.errors")
LIBRARY_CASES: tuple[tuple[type[BaseException], FailureKind], ...] = (
    (stand_in("TimeoutException", E2B_SANDBOX, module="e2b.exceptions"), FailureKind.WALL_CLOCK),
    (
        stand_in("SandboxNotFoundException", E2B_SANDBOX, module="e2b.exceptions"),
        FailureKind.SANDBOX,
    ),
    (stand_in("ServiceBusyException", module="e2b.exceptions"), FailureKind.SANDBOX),
    (stand_in("SandboxTerminatedError", module="modal.exception"), FailureKind.SANDBOX),
    (stand_in("SandboxTimeoutError", module="modal.exception"), FailureKind.WALL_CLOCK),
    (stand_in("DaytonaConnectionError", module="daytona"), FailureKind.NETWORK),
    (stand_in("DaytonaError", module="daytona"), FailureKind.SANDBOX),
    (stand_in("ContainerError", DOCKER, module="docker.errors"), FailureKind.AGENT_EXCEPTION),
    (stand_in("APIError", DOCKER, OSError, module="docker.errors"), FailureKind.SANDBOX),
    (
        stand_in("ServerDisconnectedError", AIOHTTP_CONNECTION, module="aiohttp"),
        FailureKind.NETWORK,
    ),
    (stand_in("ConnectionTimeoutError", TimeoutError, module="aiohttp"), FailureKind.NETWORK),
    (stand_in("ConnectError", HTTPX_TRANSPORT, module="httpx"), FailureKind.NETWORK),
    (
        stand_in(
            "ReadTimeout",
            stand_in("TimeoutException", HTTPX_TRANSPORT, module="httpx"),
            module="httpx",
        ),
        FailureKind.WALL_CLOCK,
    ),
    (
        stand_in("ConnectTimeout", ConnectionError, REQUESTS_TIMEOUT, module="requests.exceptions"),
        FailureKind.NETWORK,
    ),
    (
        stand_in("ReadTimeout", REQUESTS_TIMEOUT, module="requests.exceptions"),
        FailureKind.WALL_CLOCK,
    ),
    (stand_in("NewConnectionError", module="urllib3.exceptions"), FailureKind.NETWORK),
    (stand_in("OutOfMemoryError", module="ray.exceptions"), FailureKind.OOM_HOST),
    (stand_in("OutOfMemoryError", RuntimeError, module="torch"), FailureKind.OOM_GPU),
    (stand_in("ObjectStoreFullError", module="ray.exceptions"), FailureKind.OOM_HOST),
    (stand_in("ActorDiedError", RAY_ACTOR, module="ray.exceptions"), FailureKind.PROCESS_KILLED),
    (stand_in("NodeDiedError", module="ray.exceptions"), FailureKind.PROCESS_KILLED),
    (stand_in("EngineDeadError", module="vllm.v1.engine.exceptions"), FailureKind.ENGINE_CRASH),
    (stand_in("DistBackendError", RuntimeError, module="torch.distributed"), FailureKind.NETWORK),
    (
        stand_in("OverlongPromptError", PROVIDER, module="verifiers.v1.errors"),
        FailureKind.AGENT_EXCEPTION,
    ),
    (stand_in("ProviderError", module="verifiers.v1.errors"), FailureKind.ENGINE_CRASH),
    (stand_in("TaskError", module="verifiers.v1.errors"), FailureKind.GRADER_EXCEPTION),
    (
        stand_in("ToolParseError", stand_in("ToolError", module="verifiers"), module="verifiers"),
        FailureKind.MALFORMED_ACTION,
    ),
    (stand_in("InfraError", module="verifiers"), FailureKind.SANDBOX),
    (
        stand_in("RewardFileNotFoundError", FileNotFoundError, module="harbor"),
        FailureKind.MISSING_REWARD,
    ),
    (stand_in("VerifierTimeoutError", TimeoutError, module="harbor"), FailureKind.GRADER_EXCEPTION),
    (stand_in("EnvironmentStartTimeoutError", TimeoutError, module="harbor"), FailureKind.SANDBOX),
    (stand_in("RolloutTimeout", module="nemo_rl.experience.failures"), FailureKind.WALL_CLOCK),
    (stand_in("NoHealthyShards", module="nemo_rl.experience.failures"), FailureKind.ENGINE_CRASH),
    (
        stand_in("RolloutDataFailure", module="nemo_rl.experience.failures"),
        FailureKind.AGENT_EXCEPTION,
    ),
    (stand_in("InfraAbort", module="miles.rollout.agentic.agent_function"), FailureKind.SANDBOX),
)


@pytest.mark.parametrize(
    ("cls", "kind"),
    LIBRARY_CASES,
    ids=[f"{case[0].__module__}.{case[0].__name__}" for case in LIBRARY_CASES],
)
def test_library_exception_classes_classify_by_name_base_class_and_module(
    cls: type[BaseException], kind: FailureKind
) -> None:
    failure = classify_exception(cls("boom"), stage=Stage.ROLLOUT)

    assert failure.kind is kind


def test_a_ray_task_error_wrapping_an_agent_value_error_stays_the_agents() -> None:
    # Ray's as_instanceof_cause builds a class deriving from RayTaskError and the cause's class.
    ray_task_error = stand_in("RayTaskError", module="ray.exceptions")
    wrapped = stand_in(
        "RayTaskError(ValueError)", ray_task_error, ValueError, module="ray.exceptions"
    )

    assert (
        classify_exception(wrapped("bad action"), stage=Stage.ROLLOUT).kind
        is FailureKind.AGENT_EXCEPTION
    )


def test_a_class_name_alone_classifies_like_the_exception() -> None:
    assert (
        classify_error_name("SandboxError", "sandbox gone", stage=Stage.ROLLOUT).kind
        is FailureKind.SANDBOX
    )
    assert (
        classify_error_name("TaskError", "no reward.txt", stage=Stage.GRADE).kind
        is FailureKind.GRADER_EXCEPTION
    )
    unknown = classify_error_name("HarnessError", "bug", stage=Stage.ROLLOUT)
    assert unknown.kind is FailureKind.AGENT_EXCEPTION
    assert unknown.evidence == ("exception_type=HarnessError", "message=bug")


def test_an_http_status_marks_an_unknown_name_as_infrastructure_only_for_5xx_408_and_429() -> None:
    kinds = {
        status: classify_error_name("APIError", "x", stage=Stage.ROLLOUT, status_code=status).kind
        for status in (400, 404, 408, 429, 500, 503)
    }

    assert kinds == {
        400: FailureKind.AGENT_EXCEPTION,
        404: FailureKind.AGENT_EXCEPTION,
        408: FailureKind.ENGINE_CRASH,
        429: FailureKind.ENGINE_CRASH,
        500: FailureKind.ENGINE_CRASH,
        503: FailureKind.ENGINE_CRASH,
    }


def test_raise_from_none_cuts_the_chain_so_an_agent_bug_stays_the_agents() -> None:
    try:
        try:
            raise ConnectionRefusedError(111, "refused")
        except ConnectionRefusedError:
            raise ValueError("agent parse bug") from None
    except ValueError as error:
        failure = classify_exception(error, stage=Stage.ROLLOUT)

    assert failure.kind is FailureKind.AGENT_EXCEPTION


def test_an_error_raised_while_handling_another_is_classified_as_itself() -> None:
    # A KeyError in a handler for a timeout is a bug in the handler, not a timeout.
    try:
        try:
            raise TimeoutError("sandbox timed out")
        except TimeoutError:
            raise KeyError("missing field in the retry handler")  # noqa: B904 - the implicit context is the point
    except KeyError as error:
        failure = classify_exception(error, stage=Stage.ROLLOUT)

    assert failure.kind is FailureKind.AGENT_EXCEPTION


def test_a_process_that_exits_zero_is_not_a_failure_whatever_its_output_says() -> None:
    result = exec_result(
        exit_code=0, stderr="caught OutOfMemoryError, retried with a smaller batch; Killed nothing"
    )

    assert (
        classify_exec(result, resources=make_resources(), host_limit_bytes=None, stage=Stage.TOOL)
        is None
    )


def test_an_agent_value_error_that_mentions_nccl_or_xid_stays_the_agents() -> None:
    for text in (
        "the answer mentions NCCL",
        "the log parser saw Xid 79",
        "CUDA out of memory is in the prompt",
    ):
        assert (
            classify_exception(ValueError(text), stage=Stage.ROLLOUT).kind
            is FailureKind.AGENT_EXCEPTION
        ), text
