"""A final slime sample carries the decision: a masked failure is removed from the loss with status failed, a zeroed one keeps training at reward 0, and every exit status comes from the miles vocabulary."""

from __future__ import annotations

import pytest

from tests.support import make_decision, make_failure, make_record
from tplane.adapters.outcome import AdapterError
from tplane.adapters.slime import MILES_EXIT_STATUSES, exit_status, sample_fields
from tplane.schema import Action, FailureKind, Outcome, Record


def failed(kind: FailureKind, action: Action) -> Record:
    reward = 0.0 if action is Action.ZERO else None
    return make_record(
        outcome=Outcome.FAILED,
        failure=make_failure(kind=kind),
        reward=reward,
        decision=make_decision(action=action),
    )


def test_a_scored_sample_keeps_its_status_and_reward() -> None:
    fields = sample_fields(make_record(reward=0.75))

    assert fields == {
        "reward": 0.75,
        "remove_sample": False,
        "metadata": {
            "exit_status": "Submitted",
            "tp_attempt": 1,
            "tp_decision": "score",
            "tp_failure_class": None,
            "tp_failure_kind": None,
            "tp_unit_id": "unit-1",
        },
    }


def test_a_masked_infra_failure_is_removed_from_the_loss_and_marked_failed_not_aborted() -> None:
    fields = sample_fields(failed(FailureKind.SANDBOX, Action.MASK))

    assert fields["status"] == "failed"
    assert fields["remove_sample"] is True
    assert fields["reward"] == 0.0
    metadata = fields["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["exit_status"] == "SandboxUnavailable"
    assert metadata["tp_decision"] == "mask"


def test_a_zeroed_agent_failure_keeps_training_at_zero_reward_without_a_status_change() -> None:
    fields = sample_fields(failed(FailureKind.AGENT_EXCEPTION, Action.ZERO))

    assert "status" not in fields
    assert (fields["reward"], fields["remove_sample"]) == (0.0, False)


def test_every_failure_kind_maps_to_a_miles_exit_status() -> None:
    for kind in FailureKind:
        assert exit_status(kind) in MILES_EXIT_STATUSES, kind


@pytest.mark.parametrize("action", (Action.RETRY, Action.ABORT))
def test_a_retry_or_abort_record_is_not_a_final_sample(action: Action) -> None:
    with pytest.raises(
        AdapterError,
        match=f"slime samples come from final attempts; unit-1 attempt 1 was decided {action.value}",
    ):
        sample_fields(failed(FailureKind.NETWORK, action))
