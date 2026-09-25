"""Aggregate trajectory and eval records into per-source windows of a fixed number of units.

Definitions follow the MiMo RL dashboard (avg@n, avg@n_no_infra, infra_error/seq_rate):
    avg_reward          mean reward over records whose reward reached the trainer (score or zero)
    avg_reward_no_infra the same, excluding infra and grader failures, never agent failures
    zero_reward_rate    fraction of rewards that reached the trainer and were exactly 0
    infra_error_rate    infra failures / units;  grader_error_rate  grader failures / units
    poisoned_units      infra or grader failures whose zero reached the trainer
Windows count units, not wall time, because rollouts arrive in bursts.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from tplane.schema import REWARDED_KINDS, FailureClass, Record

NON_AGENT_CLASSES: Final[frozenset[FailureClass]] = frozenset(
    {FailureClass.GRADER, FailureClass.INFRA}
)


class IntegrityError(ValueError):
    """Integrity settings name a closed, valid configuration."""


@dataclass(frozen=True, kw_only=True)
class SourceWindow:
    """Reward and failure metrics for one window of consecutive units from one data source."""

    source: str | None
    index: int
    first_unit_id: str
    units: int
    avg_reward: float | None
    avg_reward_no_infra: float | None
    zero_reward_rate: float | None
    infra_error_rate: float
    grader_error_rate: float
    poisoned_units: int


def units_by_source(records: Iterable[Record]) -> list[tuple[str | None, list[Record]]]:
    """Group trajectory and eval records by source, sources sorted with None last, each in start-time order."""
    groups: dict[str | None, list[Record]] = {}
    for record in records:
        if record.kind in REWARDED_KINDS:
            groups.setdefault(record.source, []).append(record)
    ordered = sorted(groups, key=lambda source: (source is None, source or ""))
    return [(source, sorted(groups[source], key=start_order)) for source in ordered]


def start_order(record: Record) -> tuple[int, str, int]:
    """Return the key that orders records by start time, then unit and attempt."""
    return (record.started_at_ms, record.unit_id, record.attempt)


def is_non_agent_failure(record: Record) -> bool:
    """Return whether the record failed for an infra or grader cause."""
    return record.failure is not None and record.failure.failure_class in NON_AGENT_CLASSES


def is_poisoned(record: Record) -> bool:
    """Return whether an infra or grader failure's reward reached the trainer."""
    return is_non_agent_failure(record) and record.reward is not None


def source_windows(records: Iterable[Record], *, window_units: int) -> tuple[SourceWindow, ...]:
    """Split each source's units into consecutive windows of window_units; the last window may be shorter."""
    if window_units < 1:
        raise IntegrityError(f"window_units must be at least 1, got {window_units}")
    windows: list[SourceWindow] = []
    for source, units in units_by_source(records):
        for index, start in enumerate(range(0, len(units), window_units)):
            windows.append(_window(source, index, units[start : start + window_units]))
    return tuple(windows)


def _window(source: str | None, index: int, units: Sequence[Record]) -> SourceWindow:
    rewarded = [record.reward for record in units if record.reward is not None]
    clean = [
        record.reward
        for record in units
        if record.reward is not None and not is_non_agent_failure(record)
    ]
    return SourceWindow(
        source=source,
        index=index,
        first_unit_id=units[0].unit_id,
        units=len(units),
        avg_reward=_mean(rewarded),
        avg_reward_no_infra=_mean(clean),
        zero_reward_rate=None
        if len(rewarded) == 0
        else sum(value == 0.0 for value in rewarded) / len(rewarded),
        infra_error_rate=_rate(units, FailureClass.INFRA),
        grader_error_rate=_rate(units, FailureClass.GRADER),
        poisoned_units=sum(is_poisoned(record) for record in units),
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if len(values) == 0 else statistics.fmean(values)


def _rate(units: Sequence[Record], failure_class: FailureClass) -> float:
    failed = sum(
        record.failure is not None and record.failure.failure_class is failure_class
        for record in units
    )
    return failed / len(units)
