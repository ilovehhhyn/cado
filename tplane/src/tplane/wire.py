"""Render a Record to plain JSON data and parse it back without trusting the input."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final, TypeVar

from tplane.schema import (
    SCHEMA_VERSION,
    Action,
    CostLine,
    Decision,
    Failure,
    FailureKind,
    MismatchSummary,
    Outcome,
    Record,
    ResourceSummary,
    SchemaError,
    Stage,
    UnitKind,
)

RECORD_KEYS_V1: Final[frozenset[str]] = frozenset(
    {
        "attempt",
        "cost",
        "decision",
        "ended_at_ms",
        "failure",
        "kind",
        "outcome",
        "resources",
        "reward",
        "run_id",
        "schema_version",
        "source",
        "started_at_ms",
        "tags",
        "unit_id",
    }
)
RECORD_KEYS: Final[frozenset[str]] = RECORD_KEYS_V1 | {"mismatch"}
READABLE_VERSIONS: Final[tuple[int, ...]] = (1, SCHEMA_VERSION)
FAILURE_KEYS: Final[frozenset[str]] = frozenset(
    {"evidence", "exception_type", "kind", "message", "retryable", "stage"}
)
RESOURCES_KEYS: Final[frozenset[str]] = frozenset(
    {"gpu_total_bytes", "peak_gpu_used_bytes", "peak_host_rss_bytes", "sample_count"}
)
COST_KEYS: Final[frozenset[str]] = frozenset(
    {"cpu_seconds", "gpu_seconds", "tokens_in", "tokens_out", "usd"}
)
DECISION_KEYS: Final[frozenset[str]] = frozenset({"action", "attempt", "reason"})
MISMATCH_KEYS: Final[frozenset[str]] = frozenset(
    {"kl", "kl_standard_error", "ratio_mean", "ratio_standard_error", "sequences", "tokens"}
)

Member = TypeVar("Member", bound=StrEnum)


class RecordParseError(ValueError):
    """Record data must match one readable schema version exactly, with no unknown keys."""


def render_record(record: Record) -> dict[str, object]:
    """Render a record as JSON-ready data with only built-in types."""
    failure = None if record.failure is None else _render_failure(record.failure)
    return {
        "attempt": record.attempt,
        "cost": {
            "cpu_seconds": record.cost.cpu_seconds,
            "gpu_seconds": record.cost.gpu_seconds,
            "tokens_in": record.cost.tokens_in,
            "tokens_out": record.cost.tokens_out,
            "usd": record.cost.usd,
        },
        "decision": {
            "action": record.decision.action.value,
            "attempt": record.decision.attempt,
            "reason": record.decision.reason,
        },
        "ended_at_ms": record.ended_at_ms,
        "failure": failure,
        "kind": record.kind.value,
        "mismatch": None if record.mismatch is None else _render_mismatch(record.mismatch),
        "outcome": record.outcome.value,
        "resources": {
            "gpu_total_bytes": list(record.resources.gpu_total_bytes),
            "peak_gpu_used_bytes": list(record.resources.peak_gpu_used_bytes),
            "peak_host_rss_bytes": record.resources.peak_host_rss_bytes,
            "sample_count": record.resources.sample_count,
        },
        "reward": record.reward,
        "run_id": record.run_id,
        "schema_version": record.schema_version,
        "source": record.source,
        "started_at_ms": record.started_at_ms,
        "tags": [[key, value] for key, value in record.tags],
        "unit_id": record.unit_id,
    }


def parse_record(data: object) -> Record:
    """Parse untrusted JSON data into a Record or raise RecordParseError naming the offending path."""
    version = _read_version(data)
    mapping = _require_mapping(
        data, "record", RECORD_KEYS if version == SCHEMA_VERSION else RECORD_KEYS_V1
    )
    failure_data = mapping["failure"]
    failure = None if failure_data is None else _parse_failure(failure_data)
    mismatch_data = mapping.get("mismatch")
    mismatch = None if mismatch_data is None else _parse_mismatch(mismatch_data)
    try:
        return Record(
            schema_version=SCHEMA_VERSION,
            run_id=_require_str(mapping, "run_id", "record"),
            unit_id=_require_str(mapping, "unit_id", "record"),
            kind=_require_enum(mapping, "kind", "record", UnitKind),
            attempt=_require_int(mapping, "attempt", "record"),
            source=_optional_str(mapping, "source", "record"),
            started_at_ms=_require_int(mapping, "started_at_ms", "record"),
            ended_at_ms=_require_int(mapping, "ended_at_ms", "record"),
            outcome=_require_enum(mapping, "outcome", "record", Outcome),
            failure=failure,
            reward=_optional_float(mapping, "reward", "record"),
            resources=_parse_resources(mapping["resources"]),
            cost=_parse_cost(mapping["cost"]),
            decision=_parse_decision(mapping["decision"]),
            tags=_parse_tags(mapping["tags"]),
            mismatch=mismatch,
        )
    except SchemaError as error:
        raise RecordParseError(f"record: {error}") from error


def _read_version(data: object) -> int:
    """Read schema_version before the key check, because the allowed keys depend on it."""
    if type(data) is not dict:
        raise RecordParseError(f"record must be a JSON object, got {type(data).__name__}")
    if "schema_version" not in data:
        raise RecordParseError("record: missing keys ['schema_version']")
    version = _require_int(data, "schema_version", "record")
    if version not in READABLE_VERSIONS:
        raise RecordParseError(
            f"record.schema_version must be one of {list(READABLE_VERSIONS)}, got {version}"
        )
    return version


def _render_failure(failure: Failure) -> dict[str, object]:
    return {
        "evidence": list(failure.evidence),
        "exception_type": failure.exception_type,
        "kind": failure.kind.value,
        "message": failure.message,
        "retryable": failure.retryable,
        "stage": failure.stage.value,
    }


def _render_mismatch(mismatch: MismatchSummary) -> dict[str, object]:
    return {
        "kl": mismatch.kl,
        "kl_standard_error": mismatch.kl_standard_error,
        "ratio_mean": mismatch.ratio_mean,
        "ratio_standard_error": mismatch.ratio_standard_error,
        "sequences": mismatch.sequences,
        "tokens": mismatch.tokens,
    }


def _parse_mismatch(data: object) -> MismatchSummary:
    path = "record.mismatch"
    mapping = _require_mapping(data, path, MISMATCH_KEYS)
    try:
        return MismatchSummary(
            sequences=_require_int(mapping, "sequences", path),
            tokens=_require_int(mapping, "tokens", path),
            kl=_require_float(mapping, "kl", path),
            kl_standard_error=_require_float(mapping, "kl_standard_error", path),
            ratio_mean=_require_float(mapping, "ratio_mean", path),
            ratio_standard_error=_require_float(mapping, "ratio_standard_error", path),
        )
    except SchemaError as error:
        raise RecordParseError(f"{path}: {error}") from error


def _parse_failure(data: object) -> Failure:
    mapping = _require_mapping(data, "record.failure", FAILURE_KEYS)
    path = "record.failure"
    try:
        return Failure(
            kind=_require_enum(mapping, "kind", path, FailureKind),
            stage=_require_enum(mapping, "stage", path, Stage),
            retryable=_require_bool(mapping, "retryable", path),
            message=_require_str(mapping, "message", path),
            evidence=tuple(_require_str_list(mapping, "evidence", path)),
            exception_type=_optional_str(mapping, "exception_type", path),
        )
    except SchemaError as error:
        raise RecordParseError(f"{path}: {error}") from error


def _parse_resources(data: object) -> ResourceSummary:
    path = "record.resources"
    mapping = _require_mapping(data, path, RESOURCES_KEYS)
    try:
        return ResourceSummary(
            sample_count=_require_int(mapping, "sample_count", path),
            peak_host_rss_bytes=_require_int(mapping, "peak_host_rss_bytes", path),
            peak_gpu_used_bytes=tuple(_require_int_list(mapping, "peak_gpu_used_bytes", path)),
            gpu_total_bytes=tuple(_require_int_list(mapping, "gpu_total_bytes", path)),
        )
    except SchemaError as error:
        raise RecordParseError(f"{path}: {error}") from error


def _parse_cost(data: object) -> CostLine:
    path = "record.cost"
    mapping = _require_mapping(data, path, COST_KEYS)
    try:
        return CostLine(
            cpu_seconds=_require_float(mapping, "cpu_seconds", path),
            gpu_seconds=_require_float(mapping, "gpu_seconds", path),
            tokens_in=_require_int(mapping, "tokens_in", path),
            tokens_out=_require_int(mapping, "tokens_out", path),
            usd=_optional_float(mapping, "usd", path),
        )
    except SchemaError as error:
        raise RecordParseError(f"{path}: {error}") from error


def _parse_decision(data: object) -> Decision:
    path = "record.decision"
    mapping = _require_mapping(data, path, DECISION_KEYS)
    try:
        return Decision(
            action=_require_enum(mapping, "action", path, Action),
            reason=_require_str(mapping, "reason", path),
            attempt=_require_int(mapping, "attempt", path),
        )
    except SchemaError as error:
        raise RecordParseError(f"{path}: {error}") from error


def _parse_tags(data: object) -> tuple[tuple[str, str], ...]:
    if type(data) is not list:
        raise RecordParseError(f"record.tags must be a list, got {type(data).__name__}")
    tags: list[tuple[str, str]] = []
    for index, item in enumerate(data):
        is_pair = type(item) is list and len(item) == 2
        if not is_pair or any(type(part) is not str for part in item):
            raise RecordParseError(
                f"record.tags[{index}] must be a [key, value] pair of strings, got {item!r}"
            )
        tags.append((item[0], item[1]))
    return tuple(tags)


def _require_mapping(value: object, path: str, allowed: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise RecordParseError(f"{path} must be a JSON object, got {type(value).__name__}")
    keys = set(value)
    unknown = sorted(keys - allowed)
    if len(unknown) != 0:
        raise RecordParseError(f"{path}: unknown keys {unknown}")
    missing = sorted(allowed - keys)
    if len(missing) != 0:
        raise RecordParseError(f"{path}: missing keys {missing}")
    return value


def _require_int(mapping: Mapping[str, object], key: str, path: str) -> int:
    value = mapping[key]
    if type(value) is not int:
        raise RecordParseError(f"{path}.{key} must be an integer, got {value!r}")
    return value


def _require_float(mapping: Mapping[str, object], key: str, path: str) -> float:
    value = mapping[key]
    if type(value) is not float and type(value) is not int:
        raise RecordParseError(f"{path}.{key} must be a number, got {value!r}")
    return float(value)


def _optional_float(mapping: Mapping[str, object], key: str, path: str) -> float | None:
    if mapping[key] is None:
        return None
    return _require_float(mapping, key, path)


def _require_str(mapping: Mapping[str, object], key: str, path: str) -> str:
    value = mapping[key]
    if type(value) is not str:
        raise RecordParseError(f"{path}.{key} must be a string, got {value!r}")
    return value


def _optional_str(mapping: Mapping[str, object], key: str, path: str) -> str | None:
    if mapping[key] is None:
        return None
    return _require_str(mapping, key, path)


def _require_bool(mapping: Mapping[str, object], key: str, path: str) -> bool:
    value = mapping[key]
    if type(value) is not bool:
        raise RecordParseError(f"{path}.{key} must be a boolean, got {value!r}")
    return value


def _require_enum(mapping: Mapping[str, object], key: str, path: str, enum: type[Member]) -> Member:
    value = _require_str(mapping, key, path)
    choices = sorted(member.value for member in enum)
    if value not in choices:
        raise RecordParseError(f"{path}.{key} must be one of {choices}, got {value!r}")
    return enum(value)


def _require_int_list(mapping: Mapping[str, object], key: str, path: str) -> list[int]:
    value = mapping[key]
    if type(value) is not list or any(type(item) is not int for item in value):
        raise RecordParseError(f"{path}.{key} must be a list of integers, got {value!r}")
    return value


def _require_str_list(mapping: Mapping[str, object], key: str, path: str) -> list[str]:
    value = mapping[key]
    if type(value) is not list or any(type(item) is not str for item in value):
        raise RecordParseError(f"{path}.{key} must be a list of strings, got {value!r}")
    return value
