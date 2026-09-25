"""tp show renders one line per record with the failure cause, a summary of counts and lost gpu-hours, evidence for failures, and any started-but-unrecorded units."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import make_failure, make_record, make_resources
from tplane.cli.main import main
from tplane.cli.show import render_show
from tplane.schema import Action, CostLine, Decision, FailureKind, Outcome, Record, Stage, UnitKind
from tplane.store import FileRecordStore, StartedMarker

GIB = 1024 * 1024 * 1024


def oom_record(attempt: int, action: Action) -> Record:
    return make_record(
        unit_id="step-00017",
        kind=UnitKind.TRAIN_STEP,
        attempt=attempt,
        outcome=Outcome.FAILED,
        failure=make_failure(
            kind=FailureKind.OOM_GPU,
            stage=Stage.TRAIN,
            retryable=False,
            message="oom_gpu: OutOfMemoryError: CUDA out of memory",
            evidence=("exception_type=OutOfMemoryError", "message=CUDA out of memory"),
        ),
        reward=None,
        resources=make_resources(peak_gpu=(79 * GIB + GIB // 2,), total_gpu=(80 * GIB,)),
        cost=CostLine(cpu_seconds=240.0, gpu_seconds=240.0, tokens_in=0, tokens_out=0, usd=None),
        decision=Decision(action=action, reason="infra", attempt=attempt),
    )


def test_render_lists_failures_with_cause_peak_decision_and_cost() -> None:
    records = [
        make_record(unit_id="step-00016", kind=UnitKind.TRAIN_STEP),
        oom_record(1, Action.RETRY),
        oom_record(2, Action.MASK),
    ]

    lines = render_show(records, started_without_record=(), failures_only=True)

    assert lines[0].split() == [
        "unit",
        "attempt",
        "kind",
        "outcome",
        "class",
        "cause",
        "stage",
        "peak_gpu",
        "decision",
        "cost",
    ]
    assert lines[1].split() == [
        "step-00017",
        "1",
        "train_step",
        "failed",
        "infra",
        "oom_gpu",
        "train",
        "79.5/80.0",
        "GiB",
        "retry",
        "4.0",
        "gpu-min",
    ]
    assert lines[2].split()[9:] == ["mask", "4.0", "gpu-min"]
    assert (
        "failures: 2 of 3 units; infra 2, agent 0, grader 0, timeout 0; gpu-hours lost 0.13"
        in lines
    )
    assert (
        "evidence step-00017/2: exception_type=OutOfMemoryError; message=CUDA out of memory"
        in lines
    )


def test_render_without_failures_only_includes_successes_and_no_evidence_lines() -> None:
    lines = render_show(
        [make_record(unit_id="t-1")], started_without_record=(), failures_only=False
    )

    assert lines[1].split()[:4] == ["t-1", "1", "trajectory", "ok"]
    assert not any(line.startswith("evidence") for line in lines)


def test_render_reports_started_units_without_a_record() -> None:
    marker = StartedMarker(unit_id="step-00018", attempt=1, at_ms=1_700_000_000_000)

    lines = render_show([], started_without_record=(marker,), failures_only=False)

    assert (
        "started, no record: step-00018 attempt 1 at 2023-11-14T22:13:20Z; the process died before the unit finished"
        in lines
    )


def test_render_peak_gpu_is_dash_when_no_gpu_was_sampled() -> None:
    record = make_record(resources=make_resources(peak_gpu=(), total_gpu=()))

    lines = render_show([record], started_without_record=(), failures_only=False)

    assert lines[1].split()[7] == "-"


def test_main_reads_a_run_directory_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    FileRecordStore(tmp_path).append(make_record(unit_id="t-1"))

    code = main(["show", str(tmp_path)])

    assert code == 0
    assert "t-1" in capsys.readouterr().out


def test_main_rejects_a_missing_directory_with_exit_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["show", str(tmp_path / "absent")])

    assert code == 2
    assert "is not a run directory" in capsys.readouterr().err
    assert not (tmp_path / "absent").exists()


def test_main_reports_a_corrupt_record_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    FileRecordStore(tmp_path)
    (tmp_path / "records" / "bad--01.json").write_text("{}\n")

    code = main(["show", str(tmp_path)])

    assert code == 2
    assert capsys.readouterr().err.startswith("bad--01.json: record: missing keys")


def test_console_entry_runs_as_a_module(tmp_path: Path) -> None:
    FileRecordStore(tmp_path).append(make_record(unit_id="t-1"))

    completed = subprocess.run(
        [sys.executable, "-m", "tplane.cli.main", "show", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "t-1" in completed.stdout
