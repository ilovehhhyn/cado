"""Guard one rollout for a verl-style agent loop so a non-algorithm failure is masked or zeroed instead of scored.

Field names follow verl v0.9.1 `AgentLoopOutput` (verl/experimental/agent_loop/agent_loop.py); `metrics` is an
`AgentLoopMetrics` model whose fields all have defaults, so an empty dict validates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from tplane.adapters.outcome import AdapterError, outcome_fields
from tplane.schema import Record, UnitKind
from tplane.unit import Run, Unit

Output = TypeVar("Output")


def guarded_rollout(
    run: Run,
    *,
    unit_id: str,
    source: str | None,
    body: Callable[[Unit], Output],
    on_failure: Callable[[Record], Output],
) -> Output:
    """Run body inside a trajectory unit with retries; return its output, or on_failure(record) when the final attempt failed."""
    outputs: list[Output] = []

    def run_body(unit: Unit) -> None:
        outputs.append(body(unit))

    result = run.attempt(unit_id, kind=UnitKind.TRAJECTORY, body=run_body, source=source)
    if result.ok:
        return outputs[-1]
    return on_failure(result.record)


def masked_agent_loop_fields(record: Record, *, prompt_ids: Sequence[int]) -> dict[str, object]:
    """Return AgentLoopOutput-shaped fields for a failed record: no response tokens, zero reward, and the decision in extra_fields."""
    if record.failure is None:
        raise AdapterError(
            f"masked_agent_loop_fields requires a failed record, got {record.unit_id} with outcome {record.outcome.value}"
        )
    return {
        "prompt_ids": list(prompt_ids),
        "response_ids": [],
        "response_mask": [],
        "response_logprobs": [],
        "reward_score": 0.0,
        "num_turns": 0,
        "metrics": {},
        "extra_fields": outcome_fields(record),
    }
