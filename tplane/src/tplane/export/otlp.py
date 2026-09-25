"""Render records as an OTLP/JSON trace export request with OpenTelemetry GenAI attributes.

Encoding follows opentelemetry-proto v1.11.0: hex trace and span ids, 64-bit integers as decimal strings, integer
enums (kind INTERNAL = 1, status ERROR = 2). Attributes follow the GenAI conventions (semantic-conventions-genai
8ffdf56): an agent trajectory is an `invoke_agent` span whose gen_ai.conversation.id is the unit id; a failure sets
error.type and an error status, and a success leaves the status unset. Everything tplane-specific is under rl.*.
One span per record; the attempts of a unit share a trace. POST the result to a collector's /v1/traces.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, TypeAlias

from tplane import __version__
from tplane.schema import Record, UnitKind

SPAN_KIND_INTERNAL: Final[int] = 1
STATUS_CODE_ERROR: Final[int] = 2
NANOSECONDS_PER_MILLISECOND: Final[int] = 1_000_000
ID_DOMAIN: Final[bytes] = b"tplane-otlp-v1"
TRACE_ID_HEX: Final[int] = 32
SPAN_ID_HEX: Final[int] = 16
SCOPE_NAME: Final[str] = "tplane"

AttributeValue: TypeAlias = str | int | float | bool


def render_otlp(records: Sequence[Record], *, service_name: str = SCOPE_NAME) -> dict[str, object]:
    """Return an ExportTraceServiceRequest in OTLP/JSON with one span per record, in the order given."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes({"service.name": service_name})},
                "scopeSpans": [
                    {
                        "scope": {"name": SCOPE_NAME, "version": __version__},
                        "spans": [_span(record) for record in records],
                    }
                ],
            }
        ]
    }


def _span(record: Record) -> dict[str, object]:
    span: dict[str, object] = {
        "traceId": _identifier(TRACE_ID_HEX, record.run_id, record.unit_id),
        "spanId": _identifier(SPAN_ID_HEX, record.run_id, record.unit_id, str(record.attempt)),
        "name": "train_step" if record.kind is UnitKind.TRAIN_STEP else "invoke_agent",
        "kind": SPAN_KIND_INTERNAL,
        "startTimeUnixNano": str(record.started_at_ms * NANOSECONDS_PER_MILLISECOND),
        "endTimeUnixNano": str(record.ended_at_ms * NANOSECONDS_PER_MILLISECOND),
        "attributes": _attributes(_span_attributes(record)),
    }
    if record.failure is not None:
        span["status"] = {"code": STATUS_CODE_ERROR, "message": record.failure.message}
    return span


def _span_attributes(record: Record) -> dict[str, AttributeValue]:
    values: dict[str, AttributeValue] = {
        "rl.attempt": record.attempt,
        "rl.decision": record.decision.action.value,
        "rl.outcome": record.outcome.value,
        "rl.run_id": record.run_id,
        "rl.unit_id": record.unit_id,
        "rl.unit_kind": record.kind.value,
    }
    if record.kind is not UnitKind.TRAIN_STEP:
        values["gen_ai.operation.name"] = "invoke_agent"
        values["gen_ai.conversation.id"] = record.unit_id
    if record.cost.tokens_in > 0:
        values["gen_ai.usage.input_tokens"] = record.cost.tokens_in
    if record.cost.tokens_out > 0:
        values["gen_ai.usage.output_tokens"] = record.cost.tokens_out
    if record.source is not None:
        values["rl.source"] = record.source
    if record.reward is not None:
        values["rl.reward"] = record.reward
    if record.failure is not None:
        values["error.type"] = record.failure.exception_type or record.failure.kind.value
        values["rl.failure_class"] = record.failure.failure_class.value
        values["rl.failure_kind"] = record.failure.kind.value
        values["rl.failure_stage"] = record.failure.stage.value
    if record.mismatch is not None:
        values["rl.mismatch_kl"] = record.mismatch.kl
        values["rl.mismatch_ratio_mean"] = record.mismatch.ratio_mean
    return values


def _attributes(values: dict[str, AttributeValue]) -> list[dict[str, object]]:
    return [{"key": key, "value": _value(values[key])} for key in sorted(values)]


def _value(value: AttributeValue) -> dict[str, object]:
    if type(value) is bool:
        return {"boolValue": value}
    if type(value) is int:
        return {"intValue": str(value)}
    if type(value) is float:
        return {"doubleValue": value}
    return {"stringValue": value}


def _identifier(hex_length: int, *parts: str) -> str:
    """Return a stable id: the first hex_length hex digits of sha256 over a versioned domain and NUL-joined parts."""
    digest = hashlib.sha256(b"\0".join([ID_DOMAIN, *(part.encode() for part in parts)])).hexdigest()
    return digest[:hex_length]
