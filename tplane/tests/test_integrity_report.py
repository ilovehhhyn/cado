"""tp integrity prints one row per source window, then every alarm with its probability, evidence and attribution, and exits 1 when any alarm fired."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import batched_series, make_unit_record
from tplane.cli.main import main
from tplane.integrity.metrics import source_windows
from tplane.integrity.monitor import DEFAULT_INTEGRITY_OPTIONS, Alarm, Attribution, Signal, Verdict
from tplane.integrity.report import render_integrity
from tplane.schema import Action, FailureKind
from tplane.store import FileRecordStore


def test_window_rows_show_every_metric_and_a_dash_for_an_absent_average() -> None:
    records = [
        make_unit_record(0, reward=1.0),
        make_unit_record(1, failure_kind=FailureKind.SANDBOX, action=Action.MASK),
    ]
    masked_only = [
        make_unit_record(2, source="b", failure_kind=FailureKind.SANDBOX, action=Action.MASK)
    ]
    windows = source_windows(records + masked_only, window_units=50)

    lines = render_integrity(windows, (), options=DEFAULT_INTEGRITY_OPTIONS)

    assert lines[0].split() == [
        "source",
        "window",
        "first_unit",
        "units",
        "avg_reward",
        "avg_reward_no_infra",
        "zero_reward_rate",
        "infra_error_rate",
        "grader_error_rate",
        "poisoned",
    ]
    assert lines[1].split() == ["b", "0", "u-00002", "1", "-", "-", "-", "1.000", "0.000", "0"]
    assert lines[2].split() == [
        "swe",
        "0",
        "u-00000",
        "2",
        "1.000",
        "1.000",
        "0.000",
        "0.500",
        "0.000",
        "0",
    ]
    assert lines[-1] == "no alarms at P >= 0.999"


def test_alarm_lines_carry_the_probability_evidence_and_attribution() -> None:
    attribution = Attribution(
        configuration_probability=0.9995,
        failure_rate_probability=0.9995,
        mismatch_probability=None,
        verdict=Verdict.CONFIGURATION,
    )
    alarms = (
        Alarm(
            signal=Signal.POISONED,
            source="swe",
            unit_id="u-00201",
            at_ms=1,
            probability=1.0,
            detail="scored as zero",
        ),
        Alarm(
            signal=Signal.REWARD,
            source="swe",
            unit_id="u-00440",
            at_ms=2,
            probability=0.99931,
            detail="clean mean reward dropped",
            attribution=attribution,
        ),
    )

    lines = render_integrity((), alarms, options=DEFAULT_INTEGRITY_OPTIONS)

    assert (
        lines[-3]
        == "alarms at P >= 0.999 (at most 0.001 of changes are preceded by a false alarm when the model holds):"
    )
    assert lines[-2] == "poisoned  swe  u-00201  observed  scored as zero"
    assert lines[-1] == (
        "reward  swe  u-00440  P=0.9993  clean mean reward dropped; cause: configuration, "
        "P(config)=0.9995 from failure rate 0.9995 and mismatch not recorded"
    )


def test_tp_integrity_reports_alarms_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileRecordStore(tmp_path)
    for record in batched_series(
        seed=0, steps=200, change_at=100, infra_rate_after=0.5, detected_action=Action.ZERO
    ):
        store.append(record)

    code = main(["integrity", str(tmp_path)])

    output = capsys.readouterr().out
    assert code == 1
    assert "poisoned  swe" in output
    assert "failure_rate  swe" in output


def test_tp_integrity_on_a_healthy_run_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileRecordStore(tmp_path)
    for record in batched_series(seed=0, steps=40, change_at=10**9, groups=4, rollouts=2):
        store.append(record)

    code = main(["integrity", str(tmp_path), "--window-units", "100"])

    output = capsys.readouterr().out
    assert code == 0
    assert output.rstrip().endswith("no alarms at P >= 0.999")
    assert len([line for line in output.splitlines() if line.startswith("swe ")]) == 4


def test_tp_integrity_rejects_invalid_kl_levels_before_reading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["integrity", str(tmp_path / "absent"), "--healthy-kl", "0.1", "--broken-kl", "0.01"]
    )

    assert code == 2
    assert (
        "kl levels must satisfy 0 < healthy_kl < broken_kl, got 0.1 and 0.01"
        in capsys.readouterr().err
    )


def test_tp_integrity_with_unit_batching_reads_untagged_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileRecordStore(tmp_path)
    for index in range(64):
        store.append(make_unit_record(index, reward=float(index % 2)))

    code = main(["integrity", str(tmp_path), "--batching", "units"])

    assert code == 0
    assert capsys.readouterr().out.rstrip().endswith("no alarms at P >= 0.999")


def test_tp_integrity_names_the_fix_for_untagged_records_in_step_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    FileRecordStore(tmp_path).append(make_unit_record(0))

    code = main(["integrity", str(tmp_path)])

    assert code == 2
    assert (
        "tag each trajectory with its training step, or use batching=units"
        in capsys.readouterr().err
    )
