"""Classify an exception or a finished process into a typed failure, defaulting unknown causes to the agent."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Final

from tplane.schema import ExecResult, Failure, FailureKind, ResourceSummary, Stage

MAX_CAUSE_DEPTH: Final[int] = 8
MAX_EVIDENCE_CHARS: Final[int] = 200
# A SIGKILL while peak memory is within 10% of the cgroup limit is attributed to the OOM killer.
OOM_HOST_PERCENT: Final[int] = 90
SIGKILL: Final[int] = 9
INFRA_HTTP_STATUSES: Final[frozenset[int]] = frozenset({408, 429})  # plus every 5xx
RETRYABLE_KINDS: Final[frozenset[FailureKind]] = frozenset(
    {FailureKind.ENGINE_CRASH, FailureKind.NETWORK, FailureKind.PROCESS_KILLED, FailureKind.SANDBOX}
)


@dataclass(frozen=True, kw_only=True)
class ExceptionSignature:
    """One rule: an exception matches when a class in its hierarchy has type_name (from module_prefix) and its message has message_fragment."""

    type_name: str | None
    message_fragment: str | None
    kind: FailureKind
    module_prefix: str | None = None


def _named(
    kind: FailureKind, *type_names: str, module_prefix: str | None = None
) -> tuple[ExceptionSignature, ...]:
    return tuple(
        ExceptionSignature(
            type_name=name, message_fragment=None, kind=kind, module_prefix=module_prefix
        )
        for name in type_names
    )


def _fragment(
    kind: FailureKind, fragment: str, *, type_name: str | None = "RuntimeError"
) -> ExceptionSignature:
    """Match a message fragment only in a RuntimeError by default, the class torch and CUDA raise."""
    return ExceptionSignature(type_name=type_name, message_fragment=fragment, kind=kind)


# The first matching row wins, so a subclass that means something else is listed before its base class.
# Matching is by class name, never by import, so the core never loads torch, Ray or a sandbox SDK.
# Class names were read from each library's source on 2026-09-24; versions are in docs/references.md.
EXCEPTION_SIGNATURES: Final[tuple[ExceptionSignature, ...]] = (
    # Agent-side subclasses of infrastructure bases: docker ContainerError is the command failing;
    # verifiers OverlongPromptError is a clean truncation; NeMo-RL RolloutDataFailure is data.
    *_named(
        FailureKind.AGENT_EXCEPTION, "ContainerError", "OverlongPromptError", "RolloutDataFailure"
    ),
    *_named(FailureKind.MALFORMED_ACTION, "ToolError", "ModelError"),
    # Grader: Harbor reward-file errors and verifier timeout, verifiers v1 TaskError (issue #2660).
    *_named(
        FailureKind.MISSING_REWARD,
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    ),
    *_named(FailureKind.GRADER_EXCEPTION, "VerifierTimeoutError", "TaskError"),
    # Environment start-up timeouts are the sandbox's, not the agent's (Harbor).
    *_named(FailureKind.SANDBOX, "AgentSetupTimeoutError", "EnvironmentStartTimeoutError"),
    # Memory: Ray's memory monitor shares torch's class name, so it is matched by module first.
    *_named(FailureKind.OOM_HOST, "OutOfMemoryError", "ObjectStoreFullError", module_prefix="ray"),
    *_named(FailureKind.OOM_GPU, "OutOfMemoryError"),
    _fragment(FailureKind.OOM_GPU, "CUDA out of memory"),
    *_named(FailureKind.OOM_HOST, "MemoryError"),
    _fragment(FailureKind.OOM_KV_CACHE, "KV cache", type_name=None),
    # Inference engines and rollout services: vLLM, NeMo-RL, verifiers v1 provider 5xx.
    *_named(FailureKind.WALL_CLOCK, "RolloutTimeout"),
    *_named(
        FailureKind.ENGINE_CRASH,
        "EngineDeadError",
        "AsyncEngineDeadError",
        "EngineGenerateError",
        "GenerationUnavailable",
        "NoHealthyShards",
        "RolloutInfraFailure",
        "ProviderError",
        "RaySystemError",
    ),
    # Ray workers, actors and nodes that died.
    *_named(
        FailureKind.PROCESS_KILLED,
        "RayActorError",
        "WorkerCrashedError",
        "NodeDiedError",
        "LocalRayletDiedError",
        "OwnerDiedError",
        "ObjectLostError",
    ),
    # A connect timeout is a network failure: httpx and requests ConnectTimeout, urllib3, aiohttp.
    *_named(FailureKind.NETWORK, "ConnectTimeout", "ConnectTimeoutError", "ConnectionTimeoutError"),
    # Timeouts, including e2b TimeoutException (a SandboxException) and requests Timeout (an OSError).
    *_named(
        FailureKind.WALL_CLOCK,
        "TimeoutError",
        "TimeoutException",
        "Timeout",
        "ReadTimeout",
        "SandboxTimeoutError",
        "ExecTimeoutError",
        "DaytonaTimeoutError",
    ),
    _fragment(FailureKind.NETWORK, "NCCL"),
    *_named(
        FailureKind.NETWORK,
        "ConnectionError",
        "DaytonaConnectionError",
        "DistBackendError",
        "DistNetworkError",
        "DistStoreError",
        "TransportError",
        "ClientConnectionError",
        "ClientPayloadError",
        "NewConnectionError",
        "MaxRetryError",
        "NameResolutionError",
        "RpcError",
        "GymTransportError",
        "InterceptionError",
    ),
    _fragment(FailureKind.HARDWARE, "Xid"),
    _fragment(FailureKind.HARDWARE, "uncorrectable ECC"),
    # Sandbox platforms: verifiers, e2b, Modal, Daytona, docker, miles InfraAbort.
    *_named(
        FailureKind.SANDBOX,
        "SandboxError",
        "SandboxTerminatedError",
        "TunnelError",
        "InfraError",
        "EnvError",
        "SandboxException",
        "ServiceBusyException",
        "InternalFailure",
        "DaytonaError",
        "DockerException",
        "InfraAbort",
    ),
)
OOM_GPU_TEXTS: Final[tuple[str, ...]] = ("CUDA out of memory", "OutOfMemoryError")
OOM_KILLER_TEXTS: Final[tuple[str, ...]] = ("Killed", "Out of memory", "oom-kill")


def classify_exception(error: BaseException, *, stage: Stage) -> Failure:
    """Classify an exception by walking its explicit cause chain; an unmatched exception is the agent's."""
    for candidate in _exception_chain(error):
        hierarchy = tuple((cls.__module__, cls.__name__) for cls in type(candidate).__mro__)
        kind = _match(hierarchy, str(candidate))
        if kind is not None:
            return _failure_from_error(kind, stage, type(candidate).__name__, str(candidate))
    return _failure_from_error(FailureKind.AGENT_EXCEPTION, stage, type(error).__name__, str(error))


def classify_error_name(
    type_name: str, message: str, *, stage: Stage, status_code: int | None = None
) -> Failure:
    """Classify an error known only by class name, message and optional HTTP status, as framework error records carry it.

    An unknown name with status 5xx, 408 or 429 is an engine failure; this is NeMo-RL's rule for HTTP errors.
    """
    kind = _match((("", type_name),), message)
    if kind is None:
        infra_status = status_code is not None and (
            status_code >= 500 or status_code in INFRA_HTTP_STATUSES
        )
        kind = FailureKind.ENGINE_CRASH if infra_status else FailureKind.AGENT_EXCEPTION
    failure = _failure_from_error(kind, stage, type_name, message)
    if status_code is None:
        return failure
    return dataclasses.replace(
        failure, evidence=tuple(sorted({*failure.evidence, f"status_code={status_code}"}))
    )


def classify_exec(
    result: ExecResult, *, resources: ResourceSummary, host_limit_bytes: int | None, stage: Stage
) -> Failure | None:
    """Classify how a child ended; exit 0 is success whatever the output says, and a plain non-zero exit is not a failure."""
    output = result.stdout_tail + result.stderr_tail
    if result.timed_out:
        message = f"process exceeded its timeout after {result.elapsed_ms} ms"
        return _failure(
            FailureKind.WALL_CLOCK, stage, message, (f"elapsed_ms={result.elapsed_ms}",)
        )
    if result.exit_code == 0:
        return None
    if any(text in output for text in OOM_GPU_TEXTS):
        peak = f"peak_gpu_used_bytes={list(resources.peak_gpu_used_bytes)}"
        return _failure(FailureKind.OOM_GPU, stage, "process reported CUDA out of memory", (peak,))
    if result.signal is None:
        return None
    if result.signal == SIGKILL and _looks_oom_killed(output, resources, host_limit_bytes):
        evidence = (
            f"host_limit_bytes={host_limit_bytes}",
            f"peak_host_rss_bytes={resources.peak_host_rss_bytes}",
        )
        return _failure(
            FailureKind.OOM_HOST, stage, "process was killed near the host memory limit", evidence
        )
    message = f"process was killed by signal {result.signal}"
    return _failure(FailureKind.PROCESS_KILLED, stage, message, (f"signal={result.signal}",))


def _looks_oom_killed(
    output: str, resources: ResourceSummary, host_limit_bytes: int | None
) -> bool:
    near_limit = (
        host_limit_bytes is not None
        and resources.peak_host_rss_bytes * 100 >= host_limit_bytes * OOM_HOST_PERCENT
    )
    return near_limit or any(text in output for text in OOM_KILLER_TEXTS)


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and len(chain) < MAX_CAUSE_DEPTH and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        # Only an explicit cause (raise ... from ...) is followed, as NeMo-RL does: an error raised while
        # handling another is a new error, and raise ... from None cuts the chain.
        current = current.__cause__
    return tuple(chain)


def _match(hierarchy: tuple[tuple[str, str], ...], message: str) -> FailureKind | None:
    for signature in EXCEPTION_SIGNATURES:
        if _matches(signature, hierarchy, message):
            return signature.kind
    return None


def _matches(
    signature: ExceptionSignature, hierarchy: tuple[tuple[str, str], ...], message: str
) -> bool:
    prefix = signature.module_prefix
    type_matches = signature.type_name is None or any(
        name == signature.type_name and (prefix is None or module.startswith(prefix))
        for module, name in hierarchy
    )
    fragment_matches = signature.message_fragment is None or signature.message_fragment in message
    return type_matches and fragment_matches


def _failure_from_error(kind: FailureKind, stage: Stage, type_name: str, message: str) -> Failure:
    first_line = _first_line(message)
    evidence = (f"exception_type={type_name}", f"message={first_line}")
    return Failure(
        kind=kind,
        stage=stage,
        retryable=kind in RETRYABLE_KINDS,
        message=f"{kind.value}: {type_name}: {first_line}",
        evidence=tuple(sorted(set(evidence))),
        exception_type=type_name,
    )


def _failure(kind: FailureKind, stage: Stage, message: str, evidence: tuple[str, ...]) -> Failure:
    return Failure(
        kind=kind,
        stage=stage,
        retryable=kind in RETRYABLE_KINDS,
        message=f"{kind.value}: {message}",
        evidence=tuple(sorted(set(evidence))),
    )


def _first_line(text: str) -> str:
    lines = text.splitlines()
    first = lines[0] if len(lines) != 0 else ""
    return first[:MAX_EVIDENCE_CHARS]
