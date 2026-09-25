"""Decide what the trainer does with one attempt of a unit, from a closed policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, assert_never

from tplane.schema import Action, Decision, Failure, FailureClass

CLASS_ACTIONS: Final[tuple[Action, ...]] = (Action.ABORT, Action.MASK, Action.ZERO)


class PolicyError(ValueError):
    """A policy names one of abort, mask or zero per failure class and at least one attempt."""


@dataclass(frozen=True, kw_only=True)
class Policy:
    """What to do with a failed unit, by failure class, and how many attempts a retryable failure gets."""

    on_agent: Action = Action.ZERO
    on_grader: Action = Action.MASK
    on_infra: Action = Action.MASK
    on_timeout: Action = Action.ZERO
    max_attempts: int = 1

    def __post_init__(self) -> None:
        fields = (
            ("on_agent", self.on_agent),
            ("on_grader", self.on_grader),
            ("on_infra", self.on_infra),
            ("on_timeout", self.on_timeout),
        )
        choices = sorted(candidate.value for candidate in CLASS_ACTIONS)
        for name, action in fields:
            if action not in CLASS_ACTIONS:
                raise PolicyError(
                    f"{name} must be one of {choices}, got {action.value!r}; "
                    "retry is governed by max_attempts and score is reserved for success"
                )
        if self.max_attempts < 1:
            raise PolicyError(f"max_attempts must be at least 1, got {self.max_attempts}")


DEFAULT_POLICY: Final[Policy] = Policy()


def decide(failure: Failure | None, *, policy: Policy, attempt: int) -> Decision:
    """Return the decision for this attempt; retry is only returned when a later attempt exists."""
    if attempt < 1 or attempt > policy.max_attempts:
        raise PolicyError(f"attempt must be between 1 and {policy.max_attempts}, got {attempt}")
    if failure is None:
        return Decision(action=Action.SCORE, reason="unit succeeded", attempt=attempt)
    if failure.retryable and attempt < policy.max_attempts:
        reason = f"{failure.kind.value} is retryable; attempt {attempt} of {policy.max_attempts}"
        return Decision(action=Action.RETRY, reason=reason, attempt=attempt)
    field, action = _class_action(failure.failure_class, policy)
    reason = (
        f"{failure.failure_class.value} failure {failure.kind.value} at {failure.stage.value}: "
        f"{field}={action.value}"
    )
    return Decision(action=action, reason=reason, attempt=attempt)


def _class_action(failure_class: FailureClass, policy: Policy) -> tuple[str, Action]:
    match failure_class:
        case FailureClass.AGENT:
            return ("on_agent", policy.on_agent)
        case FailureClass.GRADER:
            return ("on_grader", policy.on_grader)
        case FailureClass.INFRA:
            return ("on_infra", policy.on_infra)
        case FailureClass.TIMEOUT:
            return ("on_timeout", policy.on_timeout)
        case _:
            assert_never(failure_class)
