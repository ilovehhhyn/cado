"""A record is persisted exactly once per attempt, atomically, and a start marker outlives a unit that never finished."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support import make_record
from tplane.store import (
    RECORDS_DIRECTORY,
    STARTED_DIRECTORY,
    FileRecordStore,
    StartedMarker,
    StoreError,
)


def test_append_then_iterate_returns_records_sorted_by_unit_and_attempt(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)

    store.append(make_record(unit_id="b", attempt=1))
    store.append(make_record(unit_id="a", attempt=10))
    store.append(make_record(unit_id="a", attempt=2))
    store.append(make_record(unit_id="a", attempt=1))

    assert [(record.unit_id, record.attempt) for record in store.iterate()] == [
        ("a", 1),
        ("a", 2),
        ("a", 10),
        ("b", 1),
    ]


def test_second_append_of_the_same_attempt_is_rejected_and_the_first_is_intact(
    tmp_path: Path,
) -> None:
    store = FileRecordStore(tmp_path)
    store.append(make_record(unit_id="u", reward=1.0))

    with pytest.raises(
        StoreError, match="record u--01.json already exists; a unit attempt is written exactly once"
    ):
        store.append(make_record(unit_id="u", reward=0.0))

    assert store.iterate()[0].reward == 1.0
    assert sorted(path.name for path in (tmp_path / RECORDS_DIRECTORY).iterdir()) == ["u--01.json"]


def test_record_file_is_one_line_of_sorted_json(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)

    store.append(make_record(unit_id="u"))

    text = (tmp_path / RECORDS_DIRECTORY / "u--01.json").read_text()
    assert text.endswith("\n") and text.count("\n") == 1
    assert json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")) + "\n" == text


def test_iterate_rejects_a_corrupt_file_and_names_it(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)
    (tmp_path / RECORDS_DIRECTORY / "bad--01.json").write_text('{"schema_version": 2}\n')

    with pytest.raises(StoreError, match="bad--01.json: record: missing keys"):
        store.iterate()


def test_iterate_rejects_a_truncated_file_and_names_it(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)
    (tmp_path / RECORDS_DIRECTORY / "cut--01.json").write_text('{"schema_version": 1, "run')

    with pytest.raises(StoreError, match="cut--01.json: "):
        store.iterate()


def test_iterate_ignores_temporary_files_left_by_a_crashed_write(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)
    store.append(make_record(unit_id="u"))
    (tmp_path / RECORDS_DIRECTORY / ".tmp-abandoned").write_text("{")

    assert [record.unit_id for record in store.iterate()] == ["u"]


def test_start_marker_is_listed_until_cleared(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)

    store.mark_started("step-1", 1, 5_000)
    before = store.started_without_record()
    store.clear_started("step-1", 1)
    after = store.started_without_record()

    assert before == (StartedMarker(unit_id="step-1", attempt=1, at_ms=5_000),)
    assert after == ()


def test_marker_with_a_record_is_not_reported(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)

    store.mark_started("unit-1", 1, 5_000)
    store.append(make_record(unit_id="unit-1"))

    assert store.started_without_record() == ()


def test_marking_the_same_attempt_started_twice_is_rejected(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)
    store.mark_started("step-1", 1, 5_000)

    with pytest.raises(
        StoreError,
        match="start marker step-1--01 already exists: attempt 1 of step-1 started and never wrote a record; "
        "run it as attempt 2, or delete .*step-1--01 if no process is still running it",
    ):
        store.mark_started("step-1", 1, 6_000)


def test_clearing_an_absent_marker_is_rejected(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)

    with pytest.raises(StoreError, match="start marker step-1--01 does not exist"):
        store.clear_started("step-1", 1)


def test_a_malformed_marker_is_rejected_and_named(tmp_path: Path) -> None:
    store = FileRecordStore(tmp_path)
    (tmp_path / STARTED_DIRECTORY / "step-1--01").write_text("soon\n")

    with pytest.raises(
        StoreError,
        match="start marker step-1--01 must hold an integer millisecond time, got 'soon'",
    ):
        store.started_without_record()


def test_starting_an_attempt_that_already_has_a_record_is_rejected_before_any_marker(
    tmp_path: Path,
) -> None:
    store = FileRecordStore(tmp_path)
    store.append(make_record(unit_id="u"))

    with pytest.raises(
        StoreError, match="unit u attempt 1 already has a record; it finished in an earlier run"
    ):
        store.mark_started("u", 1, 7_000)

    assert list((tmp_path / STARTED_DIRECTORY).iterdir()) == []
