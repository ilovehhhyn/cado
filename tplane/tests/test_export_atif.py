"""An ATIF trajectory gains exactly one extra.tplane block holding the record, which reads back to an equal record, and nothing else in the trajectory changes."""

from __future__ import annotations

import copy
import json

import pytest

from tests.support import make_failure, make_record
from tplane.export.atif import AtifError, annotate_atif, record_from_atif
from tplane.schema import Action, Decision, FailureKind, Outcome

# A minimal ATIF-v1.8 document, built from the field rules of Harbor RFC 0001 at tag v0.23.0.
TRAJECTORY: dict[str, object] = {
    "schema_version": "ATIF-v1.8",
    "session_id": "run-0001",
    "agent": {"name": "my-agent", "version": "0.1.0", "model_name": "qwen3-8b"},
    "steps": [
        {"step_id": 1, "source": "user", "message": "Create hello.txt"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "",
            "tool_calls": [
                {
                    "tool_call_id": "call_1",
                    "function_name": "bash",
                    "arguments": {"cmd": "echo hi > hello.txt"},
                }
            ],
            "observation": {"results": [{"source_call_id": "call_1", "content": ""}]},
            "metrics": {"prompt_tokens": 120, "completion_tokens": 30},
        },
    ],
    "final_metrics": {"total_prompt_tokens": 120, "total_completion_tokens": 30, "total_steps": 2},
    "extra": {"harness": "harbor"},
}
FAILED = make_record(
    outcome=Outcome.FAILED,
    failure=make_failure(kind=FailureKind.SANDBOX),
    reward=None,
    decision=Decision(action=Action.MASK, reason="infra", attempt=1),
)


def test_annotating_adds_only_extra_tplane_and_leaves_the_input_untouched() -> None:
    before = copy.deepcopy(TRAJECTORY)

    annotated = annotate_atif(TRAJECTORY, FAILED)

    assert TRAJECTORY == before
    extra = annotated.pop("extra")
    assert isinstance(extra, dict)
    assert sorted(extra) == ["harness", "tplane"]
    assert {key: value for key, value in before.items() if key != "extra"} == annotated


def test_the_record_round_trips_through_json() -> None:
    annotated = annotate_atif(TRAJECTORY, FAILED)

    restored = record_from_atif(json.loads(json.dumps(annotated)))

    assert restored == FAILED


def test_a_trajectory_without_extra_gains_one() -> None:
    bare = {key: value for key, value in TRAJECTORY.items() if key != "extra"}

    annotated = annotate_atif(bare, make_record())

    assert record_from_atif(annotated) == make_record()


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"schema_version": "ATIF-v2.0"},
            "schema_version must start with 'ATIF-v1.', got 'ATIF-v2.0'",
        ),
        ({"schema_version": None}, "schema_version must start with 'ATIF-v1.', got None"),
        ({"steps": []}, "an ATIF trajectory must have at least one step"),
        ({"extra": ["not", "an", "object"]}, "extra must be a JSON object, got list"),
        (
            {"extra": {"tplane": {}}},
            "trajectory already has extra.tplane; refusing to overwrite an earlier outcome",
        ),
    ),
)
def test_invalid_or_already_annotated_trajectories_are_rejected(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(AtifError, match=message):
        annotate_atif({**TRAJECTORY, **changes}, make_record())


def test_reading_a_trajectory_without_an_outcome_is_an_error() -> None:
    with pytest.raises(AtifError, match="trajectory has no extra.tplane outcome"):
        record_from_atif(TRAJECTORY)
