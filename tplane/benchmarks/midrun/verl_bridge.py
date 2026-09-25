"""Connect the benchmark hooks to the verl 0.6.1 trainer loop of rl-rewardhacking-ext, on the driver process.

The patched trainer calls rollout_overrides before generation, on_old_log_probs after the old log-probs are
recomputed, and on_rewards just before token_level_scores is set. on_rewards applies the reward fault, writes
the step's records and truth, and returns the reward tensor the trainer then uses. Settings come from environment
variables read once: BENCHMARK_DIR, BENCHMARK_RUN_ID, BENCHMARK_SEED, and the fault schedule (hooks.parse_schedule).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Final

import torch

from benchmarks.midrun.hooks import (
    FaultSchedule,
    apply_reward_fault,
    learning_rate_scale,
    parse_schedule,
    rollout_overrides,
    train_step_record,
    trajectory_records,
    write_step,
)
from tplane.mismatch import SAFETY_BOUND_LOG_RATIO, SequenceSums

MILLISECONDS_PER_SECOND: Final[int] = 1000


class BenchmarkBridge:
    """Per-run benchmark state: the fault schedule, the output directory and this step's mismatch sums."""

    def __init__(self, *, directory: Path, run_id: str, seed: int, schedule: FaultSchedule) -> None:
        self._directory = directory
        self._run_id = run_id
        self._seed = seed
        self._schedule = schedule
        self._sums: list[SequenceSums] | None = None
        self._step_started_ms = 0
        self._applied_scale = 1.0

    @classmethod
    def from_environment(cls) -> BenchmarkBridge:
        environment = dict(os.environ)
        return cls(
            directory=Path(environment["BENCHMARK_DIR"]),
            run_id=environment["BENCHMARK_RUN_ID"],
            seed=int(environment["BENCHMARK_SEED"]),
            schedule=parse_schedule(environment),
        )

    def supersede_from(self, first_step: int) -> None:
        """Move record and truth files of steps >= first_step, left by a run that crashed after its checkpoint, aside."""
        stale = sorted(
            path
            for path in self._directory.glob("*-*.jsonl")
            if path.stem.split("-")[-1].isdigit() and int(path.stem.split("-")[-1]) >= first_step
        )
        if len(stale) == 0:
            return
        target = self._directory / f"superseded-{int(time.time())}"
        target.mkdir(parents=True, exist_ok=False)
        for path in stale:
            path.rename(target / path.name)

    def rollout_overrides(self, step: int) -> dict[str, float]:
        """Start the step clock and return the rollout sampling overrides for this step."""
        self._step_started_ms = int(time.time() * MILLISECONDS_PER_SECOND)
        return rollout_overrides(self._schedule, step=step)

    def learning_rate_change(self, step: int) -> float | None:
        """Return the factor to scale the scheduler's base rates by now, or None when nothing changes."""
        target = learning_rate_scale(self._schedule, step=step)
        if target == self._applied_scale:
            return None
        factor = target / self._applied_scale
        self._applied_scale = target
        return factor

    def on_old_log_probs(self, batch: object) -> None:
        """Keep per-sequence sums of rho and k3 between rollout and recomputed old log-probs."""
        tensors = batch.batch  # type: ignore[attr-defined]  # verl DataProto
        mask = tensors["response_mask"].float()
        delta = (tensors["old_log_probs"].float() - tensors["rollout_log_probs"].float()).clamp(
            -SAFETY_BOUND_LOG_RATIO, SAFETY_BOUND_LOG_RATIO
        )
        ratio = (delta.exp() * mask).sum(-1)
        k3 = ((torch.expm1(delta) - delta).clamp_min(0.0) * mask).sum(-1)
        tokens = mask.sum(-1)
        self._sums = [
            SequenceSums(tokens=int(count), ratio_sum=float(r), k3_sum=float(k))
            for count, r, k in zip(tokens.tolist(), ratio.tolist(), k3.tolist(), strict=True)
            if int(count) > 0
        ]

    def on_rewards(self, step: int, batch: object, reward_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the reward fault, write this step's records and truth, and return the tensor the trainer uses."""
        if self._sums is None:
            raise RuntimeError(
                "on_old_log_probs must run before on_rewards; set calculate_log_probs: true"
            )
        uids = [str(uid) for uid in batch.non_tensor_batch["uid"]]  # type: ignore[attr-defined]
        mask = batch.batch["response_mask"]  # type: ignore[attr-defined]
        last = (mask.sum(-1) - 1).clamp_min(0).long()
        scores = reward_tensor.sum(-1).tolist()
        outcome = apply_reward_fault(
            self._schedule, step=step, seed=self._seed, rewards=scores, uids=uids
        )
        updated = torch.zeros_like(reward_tensor)
        updated[torch.arange(len(scores)), last] = torch.tensor(
            outcome.trainer_rewards, dtype=reward_tensor.dtype
        )
        ended_ms = int(time.time() * MILLISECONDS_PER_SECOND)
        records = trajectory_records(
            outcome,
            run_id=self._run_id,
            step=step,
            uids=uids,
            started_at_ms=self._step_started_ms,
            ended_at_ms=ended_ms,
        )
        records.append(
            train_step_record(
                run_id=self._run_id,
                step=step,
                sums=self._sums,
                started_at_ms=self._step_started_ms,
                ended_at_ms=ended_ms,
            )
        )
        write_step(self._directory, step=step, records=records, outcome=outcome)
        self._sums = None
        return updated
