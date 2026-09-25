"""A guarded rollout returns the body's output on success, the on_failure output on a masked or zeroed failure, and the masked fields carry the decision for the trainer's filter."""

from __future__ import annotations

import pytest

from tests.support import MemoryRecordStore
from tplane.adapters.outcome import AdapterError
from tplane.adapters.verl import guarded_rollout, masked_agent_loop_fields
from tplane.policy import DEFAULT_POLICY, Policy
from tplane.schema import Action, FailureKind, Record
from tplane.unit import Run, RunOptions, Unit


def make_run(policy: Policy = DEFAULT_POLICY) -> tuple[Run, MemoryRecordStore]:
    store = MemoryRecordStore()
    run = Run(
        run_id="run-1",
        store=store,
        options=RunOptions(policy=policy, sample_interval_ms=10),
        read_host=lambda: 1,
        read_gpu=lambda: (),
        gpu_total_bytes=(),
    )
    return run, store


def must_not_run(record: Record) -> dict[str, object]:
    raise AssertionError(f"on_failure must not run, got {record.unit_id}")


def test_success_returns_the_body_output_and_records_the_reward() -> None:
    run, store = make_run()

    def body(unit: Unit) -> dict[str, object]:
        unit.reward(1.0)
        return {"response_ids": [1, 2, 3], "reward_score": 1.0}

    output = guarded_rollout(run, unit_id="t-1", source="swe", body=body, on_failure=must_not_run)

    assert output == {"response_ids": [1, 2, 3], "reward_score": 1.0}
    assert store.iterate()[0].reward == 1.0


def test_sandbox_error_is_masked_and_on_failure_output_is_returned() -> None:
    run, _ = make_run()
    seen: list[Record] = []

    def body(unit: Unit) -> dict[str, object]:
        raise ConnectionRefusedError(111, "sandbox refused")

    def on_failure(record: Record) -> dict[str, object]:
        seen.append(record)
        return masked_agent_loop_fields(record, prompt_ids=[7, 8])

    output = guarded_rollout(run, unit_id="t-1", source="swe", body=body, on_failure=on_failure)

    assert seen[0].failure is not None and seen[0].failure.kind is FailureKind.NETWORK
    assert output["prompt_ids"] == [7, 8]
    assert output["response_ids"] == [] and output["response_mask"] == []
    assert output["reward_score"] == 0.0
    assert output["extra_fields"] == {
        "tp_decision": "mask",
        "tp_failure_kind": "network",
        "tp_failure_class": "infra",
        "tp_unit_id": "t-1",
        "tp_attempt": 1,
    }


def test_retry_policy_returns_the_second_attempts_output() -> None:
    run, store = make_run(Policy(max_attempts=2))
    calls = {"count": 0}

    def body(unit: Unit) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionRefusedError(111, "sandbox refused")
        unit.reward(0.5)
        return {"attempt": calls["count"]}

    output = guarded_rollout(run, unit_id="t-1", source=None, body=body, on_failure=must_not_run)

    assert output == {"attempt": 2}
    assert [record.decision.action for record in store.iterate()] == [Action.RETRY, Action.SCORE]


def test_masked_fields_reject_a_successful_record() -> None:
    run, store = make_run()

    def body(unit: Unit) -> dict[str, object]:
        unit.reward(1.0)
        return {}

    guarded_rollout(run, unit_id="t-1", source=None, body=body, on_failure=must_not_run)

    with pytest.raises(
        AdapterError,
        match="masked_agent_loop_fields requires a failed record, got t-1 with outcome ok",
    ):
        masked_agent_loop_fields(store.iterate()[0], prompt_ids=[1])
