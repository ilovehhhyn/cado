"""A decision follows the policy for the failure's class, retries only a retryable failure with attempts left, and never returns retry on the final attempt."""

from __future__ import annotations

import pytest

from tests.support import make_failure
from tplane.policy import DEFAULT_POLICY, Policy, PolicyError, decide
from tplane.schema import Action, FailureClass, FailureKind


def test_success_scores() -> None:
    decision = decide(None, policy=DEFAULT_POLICY, attempt=1)

    assert decision.action is Action.SCORE
    assert decision.attempt == 1


CLASS_CASES: tuple[tuple[FailureKind, FailureClass, str], ...] = (
    (FailureKind.AGENT_EXCEPTION, FailureClass.AGENT, "on_agent"),
    (FailureKind.MISSING_REWARD, FailureClass.GRADER, "on_grader"),
    (FailureKind.OOM_GPU, FailureClass.INFRA, "on_infra"),
    (FailureKind.WALL_CLOCK, FailureClass.TIMEOUT, "on_timeout"),
)


@pytest.mark.parametrize(
    ("kind", "failure_class", "field"), CLASS_CASES, ids=[case[2] for case in CLASS_CASES]
)
@pytest.mark.parametrize("action", (Action.ABORT, Action.MASK, Action.ZERO))
def test_non_retryable_failure_takes_the_action_for_its_class(
    kind: FailureKind, failure_class: FailureClass, field: str, action: Action
) -> None:
    policy = Policy(**{field: action}, max_attempts=3)
    failure = make_failure(kind=kind, retryable=False)

    decision = decide(failure, policy=policy, attempt=1)

    assert decision.action is action
    assert failure_class.value in decision.reason
    assert f"{field}={action.value}" in decision.reason


def test_retryable_failure_retries_until_the_last_attempt() -> None:
    policy = Policy(on_infra=Action.MASK, max_attempts=3)
    failure = make_failure(kind=FailureKind.NETWORK, retryable=True)

    actions = [decide(failure, policy=policy, attempt=attempt).action for attempt in (1, 2, 3)]

    assert actions == [Action.RETRY, Action.RETRY, Action.MASK]


def test_retryable_failure_with_single_attempt_policy_never_retries() -> None:
    failure = make_failure(kind=FailureKind.NETWORK, retryable=True)

    decision = decide(failure, policy=DEFAULT_POLICY, attempt=1)

    assert decision.action is Action.MASK


def test_policy_rejects_retry_or_score_as_a_class_action() -> None:
    for action in (Action.RETRY, Action.SCORE):
        with pytest.raises(
            PolicyError,
            match=f"on_agent must be one of \\['abort', 'mask', 'zero'\\], got '{action.value}'",
        ):
            Policy(on_agent=action)


def test_policy_rejects_zero_attempts() -> None:
    with pytest.raises(PolicyError, match="max_attempts must be at least 1, got 0"):
        Policy(max_attempts=0)


def test_decide_rejects_attempt_outside_the_policy_range() -> None:
    for attempt in (0, 2):
        with pytest.raises(PolicyError, match=f"attempt must be between 1 and 1, got {attempt}"):
            decide(None, policy=DEFAULT_POLICY, attempt=attempt)
