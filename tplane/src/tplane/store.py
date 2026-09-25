"""Persist records as one JSON file each, written atomically and never overwritten, beside start markers that outlive a killed unit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from tplane.files import FileExistsRefusal, publish_new_file, write_text
from tplane.schema import Record
from tplane.wire import RecordParseError, parse_record, render_record

RECORDS_DIRECTORY: Final[str] = "records"
STARTED_DIRECTORY: Final[str] = "started"
FILE_SUFFIX: Final[str] = ".json"
ATTEMPT_SEPARATOR: Final[str] = "--"


class StoreError(RuntimeError):
    """Every unit attempt has at most one record and at most one start marker."""


@dataclass(frozen=True, kw_only=True)
class StartedMarker:
    """A unit attempt that entered but has not written its record."""

    unit_id: str
    attempt: int
    at_ms: int


class RecordStore(Protocol):
    """Exactly-once persistence of records and start markers."""

    def append(self, record: Record) -> None: ...

    def iterate(self) -> tuple[Record, ...]: ...

    def mark_started(self, unit_id: str, attempt: int, at_ms: int) -> None: ...

    def clear_started(self, unit_id: str, attempt: int) -> None: ...

    def started_without_record(self) -> tuple[StartedMarker, ...]: ...


class FileRecordStore:
    """Record store under one run directory; safe to share between processes on a POSIX filesystem."""

    def __init__(self, root: Path) -> None:
        self._records = root / RECORDS_DIRECTORY
        self._started = root / STARTED_DIRECTORY
        self._records.mkdir(parents=True, exist_ok=True)
        self._started.mkdir(parents=True, exist_ok=True)

    def append(self, record: Record) -> None:
        """Write the record once; a second write of the same attempt is an error and changes nothing."""
        final = self._records / (_attempt_name(record.unit_id, record.attempt) + FILE_SUFFIX)
        text = json.dumps(render_record(record), sort_keys=True, separators=(",", ":")) + "\n"
        exists = f"record {final.name} already exists; a unit attempt is written exactly once"
        _write_create_exclusive(final, text, exists)

    def iterate(self) -> tuple[Record, ...]:
        """Return every record sorted by unit and attempt; a corrupt file is an error naming it."""
        records: list[Record] = []
        for path in self._records.iterdir():
            if path.name.startswith(".") or path.suffix != FILE_SUFFIX:
                continue
            try:
                records.append(parse_record(json.loads(path.read_text())))
            except (RecordParseError, json.JSONDecodeError) as error:
                raise StoreError(f"{path.name}: {error}") from error
        return tuple(sorted(records, key=lambda record: (record.unit_id, record.attempt)))

    def mark_started(self, unit_id: str, attempt: int, at_ms: int) -> None:
        """Write the start marker for one attempt, once, and only if the attempt has no record yet."""
        name = _attempt_name(unit_id, attempt)
        if (self._records / (name + FILE_SUFFIX)).exists():
            raise StoreError(finished_message(unit_id, attempt))
        marker = self._started / name
        _write_create_exclusive(
            marker, f"{at_ms}\n", stale_marker_message(unit_id, attempt, marker)
        )

    def clear_started(self, unit_id: str, attempt: int) -> None:
        """Remove the start marker of an attempt whose record is written."""
        name = _attempt_name(unit_id, attempt)
        try:
            (self._started / name).unlink()
        except FileNotFoundError as error:
            raise StoreError(f"start marker {name} does not exist") from error

    def started_without_record(self) -> tuple[StartedMarker, ...]:
        """Return the attempts that started and never wrote a record, sorted by name."""
        markers: list[StartedMarker] = []
        for path in sorted(self._started.iterdir()):
            if path.name.startswith("."):
                continue
            if (self._records / (path.name + FILE_SUFFIX)).exists():
                continue
            markers.append(_parse_marker(path))
        return tuple(markers)


def finished_message(unit_id: str, attempt: int) -> str:
    """Return the refusal for starting an attempt that already has a record."""
    return (
        f"unit {unit_id} attempt {attempt} already has a record; it finished in an earlier run, "
        "so skip it or use a new unit id"
    )


def stale_marker_message(unit_id: str, attempt: int, marker: Path) -> str:
    """Return the refusal for starting an attempt whose earlier start never wrote a record."""
    return (
        f"start marker {marker.name} already exists: attempt {attempt} of {unit_id} started and never wrote a record; "
        f"run it as attempt {attempt + 1}, or delete {marker} if no process is still running it"
    )


def _attempt_name(unit_id: str, attempt: int) -> str:
    return f"{unit_id}{ATTEMPT_SEPARATOR}{attempt:02d}"


def _parse_marker(path: Path) -> StartedMarker:
    unit_id, _, attempt_text = path.name.rpartition(ATTEMPT_SEPARATOR)
    at_text = path.read_text().strip()
    if not attempt_text.isdigit() or unit_id == "":
        raise StoreError(f"start marker {path.name} must be named <unit_id>--<attempt>")
    if not at_text.isdigit():
        raise StoreError(
            f"start marker {path.name} must hold an integer millisecond time, got {at_text!r}"
        )
    return StartedMarker(unit_id=unit_id, attempt=int(attempt_text), at_ms=int(at_text))


def _write_create_exclusive(final: Path, text: str, exists_message: str) -> None:
    try:
        publish_new_file(final, write_text(text), exists_message=exists_message)
    except FileExistsRefusal as error:
        raise StoreError(str(error)) from error
