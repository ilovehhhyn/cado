"""OTLP/JSON export gives one INTERNAL span per record, encoded exactly as the OTLP JSON rules require, with GenAI and rl.* attributes and an error status only on failure."""

from __future__ import annotations

import re

from tests.support import make_failure, make_record, make_unit_record
from tplane.export.otlp import render_otlp
from tplane.schema import (
    Action,
    CostLine,
    Decision,
    FailureKind,
    MismatchSummary,
    Outcome,
    UnitKind,
)

SPAN_KEYS = {
    "attributes",
    "endTimeUnixNano",
    "kind",
    "name",
    "spanId",
    "startTimeUnixNano",
    "status",
    "traceId",
}
VALUE_KEYS = {"boolValue", "doubleValue", "intValue", "stringValue"}


def collect(payload: dict[str, object]) -> list[dict[str, object]]:
    """In-memory collector: check the OTLP/JSON encoding rules (opentelemetry-proto v1.11.0) and decode each span."""
    assert set(payload) == {"resourceSpans"}
    (resource_spans,) = payload["resourceSpans"]
    assert decode(resource_spans["resource"]["attributes"]) == {"service.name": "tplane"}
    (scope_spans,) = resource_spans["scopeSpans"]
    assert scope_spans["scope"]["name"] == "tplane"
    spans = []
    for span in scope_spans["spans"]:
        assert set(span) <= SPAN_KEYS, set(span) - SPAN_KEYS
        assert re.fullmatch(r"[0-9a-f]{32}", span["traceId"]) and set(span["traceId"]) != {"0"}
        assert re.fullmatch(r"[0-9a-f]{16}", span["spanId"]) and set(span["spanId"]) != {"0"}
        assert span["kind"] == 1
        start, end = span["startTimeUnixNano"], span["endTimeUnixNano"]
        assert (
            type(start) is str and type(end) is str and start.isdigit() and int(start) <= int(end)
        )
        if "status" in span:
            assert set(span["status"]) == {"code", "message"} and span["status"]["code"] == 2
        spans.append({**span, "attributes": decode(span["attributes"])})
    return spans


def decode(attributes: list[dict[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for attribute in attributes:
        (kind,) = attribute["value"]
        assert kind in VALUE_KEYS
        value = attribute["value"][kind]
        if kind == "intValue":
            assert type(value) is str
            value = int(value)
        decoded[attribute["key"]] = value
    return decoded


def test_a_successful_trajectory_is_an_invoke_agent_span_without_a_status() -> None:
    record = make_record(
        unit_id="t-1",
        started_at_ms=1_700_000_000_000,
        ended_at_ms=1_700_000_005_000,
        reward=0.75,
        cost=CostLine(cpu_seconds=5.0, gpu_seconds=0.0, tokens_in=120, tokens_out=30, usd=None),
    )

    (span,) = collect(render_otlp([record]))

    assert span["name"] == "invoke_agent"
    assert "status" not in span
    assert (span["startTimeUnixNano"], span["endTimeUnixNano"]) == (
        "1700000000000000000",
        "1700000005000000000",
    )
    assert span["attributes"] == {
        "gen_ai.conversation.id": "t-1",
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 30,
        "rl.attempt": 1,
        "rl.decision": "score",
        "rl.outcome": "ok",
        "rl.reward": 0.75,
        "rl.run_id": "run-1",
        "rl.source": "swe-tasks",
        "rl.unit_id": "t-1",
        "rl.unit_kind": "trajectory",
    }


def test_a_failed_record_carries_error_type_the_failure_and_an_error_status() -> None:
    record = make_record(
        outcome=Outcome.FAILED,
        failure=make_failure(kind=FailureKind.OOM_GPU, message="oom_gpu: CUDA out of memory"),
        reward=None,
        decision=Decision(action=Action.MASK, reason="infra", attempt=1),
    )

    (span,) = collect(render_otlp([record]))

    assert span["status"] == {"code": 2, "message": "oom_gpu: CUDA out of memory"}
    attributes = span["attributes"]
    assert attributes["error.type"] == "oom_gpu"
    assert (
        attributes["rl.failure_class"],
        attributes["rl.failure_kind"],
        attributes["rl.failure_stage"],
    ) == (
        "infra",
        "oom_gpu",
        "rollout",
    )
    assert "rl.reward" not in attributes


def test_attempts_of_one_unit_share_a_trace_and_ids_are_deterministic() -> None:
    first, second = make_record(unit_id="u", attempt=1), make_record(unit_id="u", attempt=2)

    spans = collect(render_otlp([first, second]))
    again = collect(render_otlp([first, second]))
    other_run = collect(render_otlp([make_record(unit_id="u", run_id="run-2")]))

    assert spans == again
    assert spans[0]["traceId"] == spans[1]["traceId"] != other_run[0]["traceId"]
    assert spans[0]["spanId"] != spans[1]["spanId"]


def test_a_train_step_is_a_train_step_span_with_its_mismatch_and_no_genai_operation() -> None:
    summary = MismatchSummary(
        sequences=4,
        tokens=40,
        kl=1e-3,
        kl_standard_error=1e-4,
        ratio_mean=0.999,
        ratio_standard_error=1e-3,
    )
    record = make_unit_record(
        3, kind=UnitKind.TRAIN_STEP, source=None, reward=None, mismatch=summary
    )

    (span,) = collect(render_otlp([record]))

    assert span["name"] == "train_step"
    assert "gen_ai.operation.name" not in span["attributes"]
    assert (span["attributes"]["rl.mismatch_kl"], span["attributes"]["rl.mismatch_ratio_mean"]) == (
        1e-3,
        0.999,
    )
