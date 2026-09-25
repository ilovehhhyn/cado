"""Render records as a fixed-width table sorted by unit and attempt, with a summary, evidence and unfinished units."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Final

from tplane.schema import FailureClass, Outcome, Record
from tplane.store import StartedMarker

GIB: Final[int] = 1024 * 1024 * 1024
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 60 * 60
COLUMNS: Final[tuple[str, ...]] = (
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
)
SUMMARY_CLASSES: Final[tuple[FailureClass, ...]] = (
    FailureClass.INFRA,
    FailureClass.AGENT,
    FailureClass.GRADER,
    FailureClass.TIMEOUT,
)


def render_show(
    records: Sequence[Record],
    *,
    started_without_record: Sequence[StartedMarker],
    failures_only: bool,
) -> list[str]:
    """Render the table, the summary line, evidence for each failure, and started-but-unrecorded units."""
    shown = [record for record in records if not failures_only or record.outcome is Outcome.FAILED]
    rows = [COLUMNS, *[_row(record) for record in shown]]
    widths = [max(len(row[index]) for row in rows) for index in range(len(COLUMNS))]
    lines = [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    lines.append(_summary(records))
    for record in shown:
        if record.failure is not None:
            lines.append(
                f"evidence {record.unit_id}/{record.attempt}: " + "; ".join(record.failure.evidence)
            )
    for marker in started_without_record:
        lines.append(
            f"started, no record: {marker.unit_id} attempt {marker.attempt} at {_iso(marker.at_ms)}; "
            "the process died before the unit finished"
        )
    return lines


def _row(record: Record) -> tuple[str, ...]:
    failure = record.failure
    return (
        record.unit_id,
        str(record.attempt),
        record.kind.value,
        record.outcome.value,
        "-" if failure is None else failure.failure_class.value,
        "-" if failure is None else failure.kind.value,
        "-" if failure is None else failure.stage.value,
        _peak_gpu(record),
        record.decision.action.value,
        _cost(record),
    )


def _peak_gpu(record: Record) -> str:
    if len(record.resources.gpu_total_bytes) == 0:
        return "-"
    used = record.resources.peak_gpu_used_bytes[0] / GIB
    total = record.resources.gpu_total_bytes[0] / GIB
    return f"{used:.1f}/{total:.1f} GiB"


def _cost(record: Record) -> str:
    if record.cost.gpu_seconds > 0:
        return f"{record.cost.gpu_seconds / SECONDS_PER_MINUTE:.1f} gpu-min"
    return f"{record.cost.cpu_seconds / SECONDS_PER_MINUTE:.1f} cpu-min"


def _summary(records: Sequence[Record]) -> str:
    failures = [record.failure for record in records if record.failure is not None]
    counts = {failure_class: 0 for failure_class in FailureClass}
    for failure in failures:
        counts[failure.failure_class] += 1
    lost_hours = (
        sum(record.cost.gpu_seconds for record in records if record.failure is not None)
        / SECONDS_PER_HOUR
    )
    by_class = ", ".join(
        f"{failure_class.value} {counts[failure_class]}" for failure_class in SUMMARY_CLASSES
    )
    return f"failures: {len(failures)} of {len(records)} units; {by_class}; gpu-hours lost {lost_hours:.2f}"


def _iso(at_ms: int) -> str:
    return datetime.fromtimestamp(at_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
