"""A child runs from exactly the given environment, its output is captured only as a bounded tail, and a timeout kills and reaps it."""

from __future__ import annotations

import os
import sys
import time

import pytest

from tplane.exec import BASE_ENV, ExecError, exec_process


def python_env() -> dict[str, str]:
    return {**BASE_ENV, "PATH": os.environ["PATH"]}


def test_exit_code_and_tails_are_captured() -> None:
    script = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"

    result = exec_process([sys.executable, "-c", script], timeout_s=10, env=python_env())

    assert result.exit_code == 3
    assert result.signal is None
    assert result.timed_out is False
    assert result.stdout_tail == "out\n"
    assert result.stderr_tail == "err\n"


def test_child_does_not_inherit_ambient_variables() -> None:
    os.environ["TPLANE_UNDECLARED"] = "ambient"
    try:
        result = exec_process(
            [sys.executable, "-c", "import os; print(sorted(os.environ))"],
            timeout_s=10,
            env=python_env(),
        )
    finally:
        del os.environ["TPLANE_UNDECLARED"]

    assert "TPLANE_UNDECLARED" not in result.stdout_tail
    assert "'LANG'" in result.stdout_tail


def test_output_is_bounded_to_the_tail() -> None:
    result = exec_process(
        [sys.executable, "-c", "print('a' * 10_000)"],
        timeout_s=10,
        env=python_env(),
        output_tail_bytes=100,
    )

    assert len(result.stdout_tail) == 100
    assert result.stdout_tail.endswith("a\n")


def test_timeout_kills_the_child_and_reports_signal_nine() -> None:
    result = exec_process(
        [sys.executable, "-c", "import time; time.sleep(60)"], timeout_s=0.5, env=python_env()
    )

    assert result.timed_out is True
    assert result.signal == 9
    assert result.exit_code is None
    assert result.elapsed_ms < 10_000


def test_signal_termination_is_reported_as_a_signal() -> None:
    script = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"

    result = exec_process([sys.executable, "-c", script], timeout_s=10, env=python_env())

    assert result.signal == 15
    assert result.timed_out is False


def test_empty_argv_and_non_positive_timeout_are_rejected_before_spawning() -> None:
    with pytest.raises(ExecError, match="argv must name a program"):
        exec_process([], timeout_s=1, env=python_env())
    with pytest.raises(ExecError, match="timeout_s must be positive, got 0"):
        exec_process([sys.executable], timeout_s=0, env=python_env())


def test_missing_program_is_an_exec_error_naming_it() -> None:
    with pytest.raises(ExecError, match="cannot start 'tplane-no-such-program'"):
        exec_process(["tplane-no-such-program"], timeout_s=1, env=python_env())


def test_a_timeout_kills_the_whole_process_group_including_grandchildren() -> None:
    # Witness: kill() reached only the direct child, so a grandchild kept running and kept its memory.
    script = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )

    result = exec_process([sys.executable, "-c", script], timeout_s=1.0, env=python_env())

    grandchild = int(result.stdout_tail.split()[0])
    assert result.timed_out is True
    deadline = time.monotonic() + 5.0
    while not process_is_gone(grandchild):
        assert time.monotonic() < deadline, f"grandchild {grandchild} survived the timeout"
        time.sleep(0.05)


def process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False
