"""The prime-rl adapter reads real verifiers 0.3.1 Trace, Error and Reward objects, not only fakes with their field names.

Opt-in: pip install -e ".[frameworks]" and set TPLANE_FRAMEWORK_TESTS=1; any other value is an error.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest

from tests.support import MemoryRecordStore
from tplane.adapters.prime_rl import failure_from_error, record_trace
from tplane.schema import Action, FailureKind, Stage
from tplane.unit import Run, RunOptions

OPT_IN = os.environ.get("TPLANE_FRAMEWORK_TESTS")
if OPT_IN not in (None, "1"):
    raise RuntimeError(f"TPLANE_FRAMEWORK_TESTS must be 1 or unset, got {OPT_IN!r}")
pytestmark = pytest.mark.skipif(
    OPT_IN is None, reason="framework tests need TPLANE_FRAMEWORK_TESTS=1 and tplane[frameworks]"
)


def verifiers_trace() -> ModuleType:
    try:
        return importlib.import_module("verifiers.v1.trace")
    except ImportError as error:
        raise RuntimeError(
            "TPLANE_FRAMEWORK_TESTS=1 requires verifiers; pip install -e '.[frameworks]'"
        ) from error


def make_run() -> Run:
    return Run(
        run_id="run-1",
        store=MemoryRecordStore(),
        options=RunOptions(sample_interval_ms=10),
        read_host=lambda: 1,
        read_gpu=lambda: (),
        gpu_total_bytes=(),
    )


def test_a_real_ok_trace_with_real_rewards_scores_their_weighted_sum() -> None:
    trace_module = verifiers_trace()
    rewards = {
        "pass": trace_module.Reward(score=1.0, weight=0.5),
        "format": trace_module.Reward(score=0.2),
    }
    trace = trace_module.Trace.model_construct(ok=True, errors=[], rewards=rewards)

    result = record_trace(make_run(), trace, unit_id="t-1", source="swe")

    assert result.record.decision.action is Action.SCORE
    assert result.record.reward == pytest.approx(0.7, rel=1e-12, abs=0.0)


def test_a_real_ok_trace_whose_scoring_did_not_run_is_a_missing_reward() -> None:
    trace_module = verifiers_trace()
    trace = trace_module.Trace.model_construct(ok=True, errors=[], rewards={"solved": None})

    result = record_trace(make_run(), trace, unit_id="t-1", source="swe")

    assert (
        result.record.failure is not None
        and result.record.failure.kind is FailureKind.MISSING_REWARD
    )


def test_a_real_error_record_with_an_http_status_classifies_as_an_engine_failure() -> None:
    trace_module = verifiers_trace()
    error = trace_module.Error(type="ProviderError", message="bad gateway", status_code=502)

    failure = failure_from_error(error, stage=Stage.ROLLOUT)

    assert failure.kind is FailureKind.ENGINE_CRASH
    assert "status_code=502" in failure.evidence
