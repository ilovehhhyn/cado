"""tp export writes a run's records as OTLP/JSON or Parquet to a new file and refuses to replace an existing one."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.support import make_record
from tplane.cli.main import main
from tplane.store import FileRecordStore


def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "run"
    store = FileRecordStore(directory)
    store.append(make_record(unit_id="a"))
    store.append(make_record(unit_id="b"))
    return directory


def test_export_otlp_writes_one_span_per_record(tmp_path: Path) -> None:
    output = tmp_path / "traces.json"

    code = main(["export", "otlp", str(run_dir(tmp_path)), "--output", str(output)])

    payload = json.loads(output.read_text())
    assert code == 0
    assert len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2


def test_export_parquet_writes_one_row_per_record(tmp_path: Path) -> None:
    output = tmp_path / "records.parquet"

    code = main(["export", "parquet", str(run_dir(tmp_path)), "--output", str(output)])

    assert code == 0
    assert pq.read_table(output).column("unit_id").to_pylist() == ["a", "b"]


@pytest.mark.parametrize("export_format", ("otlp", "parquet"))
def test_export_refuses_to_replace_an_existing_file(
    export_format: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out"
    output.write_text("keep me")

    code = main(["export", export_format, str(run_dir(tmp_path)), "--output", str(output)])

    assert code == 2
    assert "out already exists; tp never replaces an export" in capsys.readouterr().err
    assert output.read_text() == "keep me"


def test_export_of_a_missing_run_directory_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "traces.json"

    code = main(["export", "otlp", str(tmp_path / "absent"), "--output", str(output)])

    assert code == 2
    assert "is not a run directory" in capsys.readouterr().err
    assert not output.exists()


def test_an_output_in_a_missing_directory_is_a_message_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "export",
            "otlp",
            str(run_dir(tmp_path)),
            "--output",
            str(tmp_path / "missing" / "traces.json"),
        ]
    )

    assert code == 2
    assert "No such file or directory" in capsys.readouterr().err
