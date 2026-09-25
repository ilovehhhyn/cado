"""Map a final tplane record onto the fields of a slime Sample, with the miles exit-status vocabulary.

Fields follow slime v0.3.2 (slime/utils/types.py): `reward`, `remove_sample` (slime zeroes the loss mask) and
`status`, whose value "failed" is the only failure status. "aborted" means a partial rollout to resume, so it is
never used for a failure. `metadata["exit_status"]` uses the vocabulary of miles PR #2802 (open when read).
The trainer must still drop removed samples from the group baseline; slime computes the baseline over the group.
"""

from __future__ import annotations

from typing import Final, assert_never

from tplane.adapters.outcome import AdapterError, outcome_fields
from tplane.schema import Action, FailureKind, Record

SUBMITTED: Final[str] = "Submitted"
MILES_EXIT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "AgentError",
        "AgentSetupFailed",
        "NonCanonicalVerifier",
        "SandboxUnavailable",
        "SequenceLengthLimitExceeded",
        "ServerUnreachable",
        SUBMITTED,
        "TimeLimitExceeded",
        "VerifierError",
    }
)
FAILED_STATUS: Final[str] = "failed"


def sample_fields(record: Record) -> dict[str, object]:
    """Return the Sample fields to set for a final attempt: reward, remove_sample, metadata, and status when failed."""
    action = record.decision.action
    if action is Action.RETRY or action is Action.ABORT:
        raise AdapterError(
            f"slime samples come from final attempts; {record.unit_id} attempt {record.attempt} was decided {action.value}"
        )
    status = SUBMITTED if record.failure is None else exit_status(record.failure.kind)
    fields: dict[str, object] = {
        "reward": 0.0 if record.reward is None else record.reward,
        "remove_sample": action is Action.MASK,
        "metadata": {"exit_status": status, **outcome_fields(record)},
    }
    if action is Action.MASK:
        fields["status"] = FAILED_STATUS
    return fields


def exit_status(kind: FailureKind) -> str:
    """Return the nearest miles exit status for a failure kind; tp_failure_kind keeps the exact kind."""
    match kind:
        case FailureKind.AGENT_EXCEPTION | FailureKind.MALFORMED_ACTION:
            return "AgentError"
        case FailureKind.WALL_CLOCK:
            return "TimeLimitExceeded"
        case FailureKind.GRADER_EXCEPTION | FailureKind.MISSING_REWARD:
            return "VerifierError"
        case (
            FailureKind.SANDBOX
            | FailureKind.OOM_HOST
            | FailureKind.PROCESS_KILLED
            | FailureKind.HARDWARE
        ):
            return "SandboxUnavailable"
        case (
            FailureKind.NETWORK
            | FailureKind.ENGINE_CRASH
            | FailureKind.OOM_GPU
            | FailureKind.OOM_KV_CACHE
        ):
            return "ServerUnreachable"
        case _:
            assert_never(kind)
