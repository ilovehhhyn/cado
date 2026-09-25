"""The analysis stitches a branch onto its parent's history, scores every desired behaviour, and counts only post-onset alarms as detections."""

from __future__ import annotations

import random
from pathlib import Path

from benchmarks.midrun.analyze import Arm, RunSpec, evaluate_run, load_run
from benchmarks.midrun.hooks import (
    Fault,
    FaultSchedule,
    apply_reward_fault,
    train_step_record,
    trajectory_records,
    write_step,
)
from tplane.mismatch import SequenceSums

GROUPS, ROLLOUTS = 8, 4


def write_run(
    directory: Path, *, steps: range, schedule: FaultSchedule, seed: int, kl: float = 4e-4
) -> None:
    rng = random.Random(seed)
    for step in steps:
        uids = [f"p{step}-{group}" for group in range(GROUPS) for _ in range(ROLLOUTS)]
        rewards = []
        for _group in range(GROUPS):
            difficulty = min(1.0, max(0.0, rng.gauss(0.5, 0.25)))
            rewards += [1.0 if rng.random() < difficulty else 0.0 for _ in range(ROLLOUTS)]
        outcome = apply_reward_fault(schedule, step=step, seed=seed, rewards=rewards, uids=uids)
        records = trajectory_records(
            outcome,
            run_id="run-1",
            step=step,
            uids=uids,
            started_at_ms=step * 1_000,
            ended_at_ms=step * 1_000 + 500,
        )
        sums = [
            SequenceSums(tokens=100, ratio_sum=100.0 * rng.gauss(1.0, 0.001), k3_sum=100.0 * kl)
            for _ in range(GROUPS * ROLLOUTS)
        ]
        records.append(
            train_step_record(
                run_id="run-1",
                step=step,
                sums=sums,
                started_at_ms=step * 1_000,
                ended_at_ms=step * 1_000 + 500,
            )
        )
        write_step(directory, step=step, records=records, outcome=outcome)


NONE = FaultSchedule(fault=Fault.NONE, start=0, rate=0.0, typed=0.0)


def test_a_branch_is_stitched_after_its_parents_pre_fork_steps(tmp_path: Path) -> None:
    write_run(tmp_path / "parent", steps=range(1, 121), schedule=NONE, seed=1)
    write_run(tmp_path / "branch", steps=range(101, 161), schedule=NONE, seed=2)
    spec = RunSpec(
        name="n1",
        arm=Arm.NULL,
        directory=tmp_path / "branch",
        parent=tmp_path / "parent",
        fork_step=100,
        onset=101,
    )

    records, truth = load_run(spec)

    steps = sorted({int(dict(record.tags)["step"]) for record in records})
    assert steps == list(range(1, 161))
    assert len(truth) == 160 * GROUPS * ROLLOUTS


def test_a_healthy_run_passes_no_overflagging_and_a_zeroed_outage_is_seen_at_once(
    tmp_path: Path,
) -> None:
    write_run(tmp_path / "parent", steps=range(1, 101), schedule=NONE, seed=1)
    zeroed = FaultSchedule(fault=Fault.SANDBOX_ZEROED, start=101, rate=0.3, typed=1.0)
    write_run(tmp_path / "branch", steps=range(101, 141), schedule=zeroed, seed=3)
    healthy = RunSpec(
        name="h1",
        arm=Arm.HEALTHY,
        directory=tmp_path / "parent",
        parent=None,
        fork_step=None,
        onset=None,
    )
    faulty = RunSpec(
        name="f1c",
        arm=Arm.SANDBOX_ZEROED,
        directory=tmp_path / "branch",
        parent=tmp_path / "parent",
        fork_step=100,
        onset=101,
    )

    healthy_result = evaluate_run(healthy)
    faulty_result = evaluate_run(faulty)

    assert healthy_result.checks == {"D1 no alarm on healthy steps": True}
    assert faulty_result.checks["D2 poisoned at the first zeroed failure"] is True
    assert faulty_result.checks["D1 no alarm before onset"] is True
    assert (
        faulty_result.delays["failure_rate"] is not None
        and faulty_result.delays["failure_rate"] <= 3
    )


def test_a_mismatch_arm_fails_its_check_when_no_mismatch_is_injected(tmp_path: Path) -> None:
    # Control: the check must be able to fail, so a mismatch arm with healthy kl does not pass D7.
    write_run(tmp_path / "parent", steps=range(1, 101), schedule=NONE, seed=1)
    write_run(tmp_path / "branch", steps=range(101, 141), schedule=NONE, seed=4)
    spec = RunSpec(
        name="f3b",
        arm=Arm.TEMPERATURE,
        directory=tmp_path / "branch",
        parent=tmp_path / "parent",
        fork_step=100,
        onset=101,
    )

    result = evaluate_run(spec)

    assert result.checks["D7 mismatch raised after onset"] is False
