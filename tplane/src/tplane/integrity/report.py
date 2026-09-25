"""Render per-source windows and integrity alarms as fixed-width text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from tplane.integrity.metrics import SourceWindow
from tplane.integrity.monitor import Alarm, Attribution, IntegrityOptions, Signal

COLUMNS: Final[tuple[str, ...]] = (
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
)


def render_integrity(
    windows: Sequence[SourceWindow], alarms: Sequence[Alarm], *, options: IntegrityOptions
) -> list[str]:
    """Render the window table, then one line per alarm or a line saying there were none."""
    lines = _table([COLUMNS, *[_row(window) for window in windows]])
    threshold = f"P >= {options.alarm_probability}"
    if len(alarms) == 0:
        return [*lines, f"no alarms at {threshold}"]
    alpha = round(1 - options.alarm_probability, 12)
    lines.append(
        f"alarms at {threshold} (at most {alpha} of changes are preceded by a false alarm when the model holds):"
    )
    return [*lines, *[_alarm_line(alarm) for alarm in alarms]]


def _row(window: SourceWindow) -> tuple[str, ...]:
    return (
        "-" if window.source is None else window.source,
        str(window.index),
        window.first_unit_id,
        str(window.units),
        _number(window.avg_reward),
        _number(window.avg_reward_no_infra),
        _number(window.zero_reward_rate),
        _number(window.infra_error_rate),
        _number(window.grader_error_rate),
        str(window.poisoned_units),
    )


def _alarm_line(alarm: Alarm) -> str:
    source = "-" if alarm.source is None else alarm.source
    probability = "observed" if alarm.signal is Signal.POISONED else f"P={alarm.probability:.4f}"
    line = f"{alarm.signal.value}  {source}  {alarm.unit_id}  {probability}  {alarm.detail}"
    if alarm.attribution is None:
        return line
    return f"{line}; {_cause(alarm.attribution)}"


def _cause(attribution: Attribution) -> str:
    mismatch = (
        "mismatch not recorded"
        if attribution.mismatch_probability is None
        else f"mismatch {attribution.mismatch_probability:.4f}"
    )
    return (
        f"cause: {attribution.verdict.value}, P(config)={attribution.configuration_probability:.4f} "
        f"from failure rate {attribution.failure_rate_probability:.4f} and {mismatch}"
    )


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _table(rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in rows
    ]
