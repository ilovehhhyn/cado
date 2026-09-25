"""A rendered record parses back to an equal record, and any deviation from the schema is rejected before a Record exists."""

from __future__ import annotations

import json

import pytest

from tests.support import make_failure, make_record
from tplane.schema import SCHEMA_VERSION, Action, Decision, FailureKind, MismatchSummary, Outcome
from tplane.wire import RecordParseError, parse_record, render_record


def test_render_then_parse_returns_an_equal_record_for_success_and_failure() -> None:
    failed = make_record(
        outcome=Outcome.FAILED,
        failure=make_failure(kind=FailureKind.OOM_GPU, evidence=("a=1", "b=2")),
        reward=None,
        decision=Decision(action=Action.MASK, reason="masked", attempt=1),
        tags=(("gpu_devices", "1"),),
    )
    for record in (make_record(), failed):
        rendered = render_record(record)

        parsed = parse_record(json.loads(json.dumps(rendered)))

        assert parsed == record


def test_render_is_deterministic_json() -> None:
    record = make_record()

    first = json.dumps(render_record(record), sort_keys=True)
    second = json.dumps(render_record(record), sort_keys=True)

    assert first == second


def test_parse_rejects_unknown_top_level_key_and_names_it() -> None:
    data = render_record(make_record())
    data["extra"] = 1

    with pytest.raises(RecordParseError, match="record: unknown keys \\['extra'\\]"):
        parse_record(data)


def test_parse_rejects_missing_key_and_names_it() -> None:
    data = render_record(make_record())
    del data["reward"]

    with pytest.raises(RecordParseError, match="record: missing keys \\['reward'\\]"):
        parse_record(data)


def test_parse_rejects_wrong_schema_version() -> None:
    data = render_record(make_record())
    data["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(
        RecordParseError,
        match=f"record.schema_version must be one of \\[1, 2\\], got {SCHEMA_VERSION + 1}",
    ):
        parse_record(data)


def test_parse_rejects_bool_where_int_is_required() -> None:
    data = render_record(make_record())
    data["attempt"] = True

    with pytest.raises(RecordParseError, match="record.attempt must be an integer, got True"):
        parse_record(data)


def test_parse_rejects_unknown_enum_value() -> None:
    data = render_record(make_record())
    data["kind"] = "banana"

    with pytest.raises(
        RecordParseError,
        match="record.kind must be one of \\['eval', 'train_step', 'trajectory'\\], got 'banana'",
    ):
        parse_record(data)


def test_parse_rejects_non_mapping_input() -> None:
    with pytest.raises(RecordParseError, match="record must be a JSON object, got list"):
        parse_record([])


def test_parse_reports_nested_path_for_failure_fields() -> None:
    data = render_record(make_record(outcome=Outcome.FAILED, failure=make_failure(), reward=None))
    failure = data["failure"]
    assert isinstance(failure, dict)
    failure["retryable"] = "yes"

    with pytest.raises(
        RecordParseError, match="record.failure.retryable must be a boolean, got 'yes'"
    ):
        parse_record(data)


def test_parse_rejects_a_malformed_tag_pair() -> None:
    data = render_record(make_record())
    data["tags"] = [["only-key"]]

    with pytest.raises(
        RecordParseError, match="record.tags\\[0\\] must be a \\[key, value\\] pair of strings"
    ):
        parse_record(data)


def test_parse_wraps_a_domain_rule_violation_with_the_record_path() -> None:
    data = render_record(make_record())
    data["unit_id"] = "../escape"

    with pytest.raises(RecordParseError, match="record: unit_id must match"):
        parse_record(data)


MISMATCH = MismatchSummary(
    sequences=8,
    tokens=800,
    kl=2.5e-4,
    kl_standard_error=1e-5,
    ratio_mean=0.999,
    ratio_standard_error=2e-3,
)


def test_a_record_with_a_mismatch_block_round_trips() -> None:
    record = make_record(mismatch=MISMATCH)

    parsed = parse_record(json.loads(json.dumps(render_record(record))))

    assert parsed == record
    assert parsed.mismatch == MISMATCH


def test_a_version_1_document_parses_into_the_current_record_without_mismatch() -> None:
    # Invariant: records written by tplane 0.1 schema version 1 stay readable after the version 2 bump.
    data = render_record(make_record())
    data["schema_version"] = 1
    del data["mismatch"]

    parsed = parse_record(data)

    assert parsed.schema_version == SCHEMA_VERSION
    assert parsed.mismatch is None
    assert parsed == make_record()


def test_a_version_1_document_with_a_mismatch_key_is_rejected() -> None:
    data = render_record(make_record())
    data["schema_version"] = 1

    with pytest.raises(RecordParseError, match="record: unknown keys \\['mismatch'\\]"):
        parse_record(data)


def test_a_version_2_document_without_the_mismatch_key_is_rejected() -> None:
    data = render_record(make_record())
    del data["mismatch"]

    with pytest.raises(RecordParseError, match="record: missing keys \\['mismatch'\\]"):
        parse_record(data)


def test_parse_reports_nested_path_for_mismatch_fields() -> None:
    data = render_record(make_record(mismatch=MISMATCH))
    block = data["mismatch"]
    assert isinstance(block, dict)
    block["sequences"] = 1.5

    with pytest.raises(
        RecordParseError, match="record.mismatch.sequences must be an integer, got 1.5"
    ):
        parse_record(data)
