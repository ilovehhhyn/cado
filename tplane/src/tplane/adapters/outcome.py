"""Render the outcome fields every adapter attaches to its framework's sample, and the adapter error."""

from __future__ import annotations

from tplane.schema import Record


class AdapterError(ValueError):
    """An adapter converts only the records its framework contract allows."""


def outcome_fields(record: Record) -> dict[str, object]:
    """Return the tp_* fields a trainer filter reads: the decision, the failure class and kind, the unit and attempt."""
    failure = record.failure
    return {
        "tp_attempt": record.attempt,
        "tp_decision": record.decision.action.value,
        "tp_failure_class": None if failure is None else failure.failure_class.value,
        "tp_failure_kind": None if failure is None else failure.kind.value,
        "tp_unit_id": record.unit_id,
    }
