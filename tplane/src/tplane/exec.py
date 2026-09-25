"""Run one child process from an explicit environment with a timeout and bounded output capture."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Final, Protocol

from tplane.schema import ExecResult

BASE_ENV: Final[dict[str, str]] = {"LANG": "C", "LC_ALL": "C", "NO_COLOR": "1", "TZ": "UTC"}
DEFAULT_OUTPUT_TAIL_BYTES: Final[int] = 64 * 1024
KILL_GRACE_S: Final[float] = 5.0


class ExecError(RuntimeError):
    """A child can only be started from a non-empty argv with a positive timeout."""


class ExecFunction(Protocol):
    """The shape of exec_process, so a backend or a test can stand in for it."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str],
        cwd: Path | None = None,
    ) -> ExecResult: ...


def monotonic_ms() -> int:
    """Return a monotonic clock reading in milliseconds."""
    return int(time.monotonic() * 1000)


def exec_process(
    argv: Sequence[str],
    *,
    timeout_s: float,
    env: Mapping[str, str],
    cwd: Path | None = None,
    output_tail_bytes: int = DEFAULT_OUTPUT_TAIL_BYTES,
    clock_ms: Callable[[], int] = monotonic_ms,
) -> ExecResult:
    """Run argv to completion or kill it at the timeout; capture only the last output_tail_bytes of each stream."""
    if len(argv) == 0:
        raise ExecError("argv must name a program")
    if timeout_s <= 0:
        raise ExecError(f"timeout_s must be positive, got {timeout_s}")
    started = clock_ms()
    # Output goes to files, not pipes, because communicate() would buffer unbounded output in memory.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            # A new session makes the child lead its own process group, so a timeout can kill its descendants too.
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                env=dict(env),
                cwd=cwd,
                start_new_session=True,
            )
        except OSError as error:
            raise ExecError(f"cannot start {argv[0]!r}: {error.strerror}") from error
        timed_out = _wait_or_kill(process, timeout_s)
        elapsed_ms = clock_ms() - started
        stdout_tail = _read_tail(out, output_tail_bytes)
        stderr_tail = _read_tail(err, output_tail_bytes)
    returncode = process.returncode
    signal_number = -returncode if returncode < 0 else None
    exit_code = returncode if returncode >= 0 else None
    return ExecResult(
        exit_code=exit_code,
        signal=signal_number,
        timed_out=timed_out,
        elapsed_ms=elapsed_ms,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _wait_or_kill(process: subprocess.Popen[bytes], timeout_s: float) -> bool:
    """Wait for the child; on timeout, or if the wait is interrupted, SIGKILL its whole process group and reap it."""
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        return True
    except BaseException:
        _kill_group(process)
        raise
    return False


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # the group already exited between the timeout and the kill
    try:
        process.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired as error:
        raise ExecError(
            f"process {process.pid} did not exit {KILL_GRACE_S} s after SIGKILL to its group"
        ) from error


def _read_tail(handle: IO[bytes], tail_bytes: int) -> str:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - tail_bytes))
    return handle.read().decode("utf-8", errors="replace")
