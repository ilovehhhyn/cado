"""Score every mid-run benchmark run against the desired behaviours D1-D8 of the benchmark plan, from its records and truth.

A fork branch is stitched after its parent's steps up to the fork, so the monitor sees the history a deployment
would. Only an alarm at or after the fault's onset step counts as a detection; any earlier alarm is a false alarm.
Run: python analyze.py --manifest runs.json --output docs/results/midrun-benchmark.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from tplane.integrity.monitor import Alarm, Signal, Verdict, detect_alarms
from tplane.schema import Record
from tplane.wire import parse_record


class Arm(StrEnum):
    """The arms of the benchmark plan."""

    GRADER_MISSING = "grader_missing"
    HEALTHY = "healthy"
    LOOPHOLE = "loophole"
    LR_SPIKE = "lr_spike"
    NULL = "null"
    SANDBOX_MASKED = "sandbox_masked"
    SANDBOX_PARTLY_TYPED = "sandbox_partly_typed"
    SANDBOX_UNTYPED = "sandbox_untyped"
    SANDBOX_ZEROED = "sandbox_zeroed"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"


@dataclass(frozen=True, kw_only=True)
class RunSpec:
    """One run: its arm, its records directory, and for a fork its parent and fork step and the fault onset."""

    name: str
    arm: Arm
    directory: Path
    parent: Path | None
    fork_step: int | None
    onset: int | None


@dataclass(frozen=True, kw_only=True)
class RunResult:
    """The checks a run passed or failed, its alarm delays in steps after onset, and its alarms."""

    name: str
    arm: Arm
    steps: int
    checks: dict[str, bool]
    delays: dict[str, int | None]
    alarms: list[str] = field(default_factory=list)


def load_run(spec: RunSpec) -> tuple[list[Record], list[dict[str, object]]]:
    """Return the run's records and truth rows, the parent's steps up to the fork first."""
    parts = [(spec.directory, None)]
    if spec.parent is not None:
        parts.insert(0, (spec.parent, spec.fork_step))
    records: list[Record] = []
    truth: list[dict[str, object]] = []
    for directory, last_step in parts:
        for path in sorted(directory.glob("records-*.jsonl")):
            step = int(path.stem.split("-")[1])
            if last_step is not None and step > last_step:
                continue
            records += [parse_record(json.loads(line)) for line in path.read_text().splitlines()]
            truth_path = directory / f"truth-{step:05d}.jsonl"
            truth += [json.loads(line) for line in truth_path.read_text().splitlines()]
    return records, truth


def evaluate_run(spec: RunSpec) -> RunResult:
    """Run the monitor on the run and score the behaviours that apply to its arm."""
    records, _ = load_run(spec)
    alarms = detect_alarms(records)
    steps = len({dict(record.tags)["step"] for record in records})
    onset = spec.onset if spec.onset is not None else 10**9
    early = [alarm for alarm in alarms if _step(alarm, records) < onset]
    late = [alarm for alarm in alarms if _step(alarm, records) >= onset]
    first = {
        signal.value: next((alarm for alarm in late if alarm.signal is signal), None)
        for signal in Signal
    }
    delays = {
        name: None if alarm is None else _step(alarm, records) - onset
        for name, alarm in first.items()
    }
    checks: dict[str, bool] = {}
    if spec.arm in (Arm.HEALTHY, Arm.NULL):
        checks["D1 no alarm on healthy steps"] = len(alarms) == 0
    else:
        checks["D1 no alarm before onset"] = len(early) == 0
    reward = first["reward"]
    verdict = None if reward is None or reward.attribution is None else reward.attribution.verdict
    match spec.arm:
        case Arm.SANDBOX_ZEROED:
            zeroed = next(
                record for record in records if record.failure is not None and record.reward == 0.0
            )
            poisoned = first["poisoned"]
            checks["D2 poisoned at the first zeroed failure"] = (
                poisoned is not None and poisoned.unit_id == zeroed.unit_id
            )
            checks["D3 failure rate raised"] = first["failure_rate"] is not None
        case Arm.SANDBOX_MASKED | Arm.GRADER_MISSING:
            checks["D2 never poisoned"] = first["poisoned"] is None
            checks["D3 failure rate raised"] = first["failure_rate"] is not None
        case Arm.SANDBOX_PARTLY_TYPED:
            checks["D3 failure rate raised"] = first["failure_rate"] is not None
            checks["D4 a reward alarm, if any, is attributed to configuration"] = verdict in (
                None,
                Verdict.CONFIGURATION,
            )
        case Arm.SANDBOX_UNTYPED:
            checks["recorded limit: verdict is not configuration"] = (
                verdict is not Verdict.CONFIGURATION
            )
        case Arm.TOP_P | Arm.TEMPERATURE:
            checks["D7 mismatch raised after onset"] = first["mismatch"] is not None
            checks["D4 a reward alarm, if any, is attributed to configuration"] = verdict in (
                None,
                Verdict.CONFIGURATION,
            )
        case Arm.LR_SPIKE:
            checks["D5 a reward alarm is attributed to the algorithm"] = (
                verdict is Verdict.ALGORITHM
            )
            checks["D7 no mismatch alarm"] = first["mismatch"] is None
        case Arm.LOOPHOLE:
            config_signals = [
                alarm
                for alarm in alarms
                if alarm.signal in (Signal.FAILURE_RATE, Signal.POISONED, Signal.MISMATCH)
            ]
            checks["D6 no infra, poisoning or mismatch alarm"] = len(config_signals) == 0
            checks["D6 never a configuration verdict"] = verdict is not Verdict.CONFIGURATION
        case Arm.HEALTHY | Arm.NULL:
            pass
    return RunResult(
        name=spec.name,
        arm=spec.arm,
        steps=steps,
        checks=checks,
        delays=delays,
        alarms=[f"{alarm.signal.value}@{_step(alarm, records)}" for alarm in alarms],
    )


def render(results: list[RunResult]) -> list[str]:
    """Render the per-run table and the false-alarm summary."""
    lines = [
        "| run | arm | steps | alarms (signal@step) | checks passed | failed checks |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        failed = [name for name, passed in result.checks.items() if not passed]
        lines.append(
            f"| {result.name} | {result.arm.value} | {result.steps} | {' '.join(result.alarms) or '-'} | "
            f"{sum(result.checks.values())}/{len(result.checks)} | {'; '.join(failed) or '-'} |"
        )
    return lines


def _step(alarm: Alarm, records: list[Record]) -> int:
    by_unit = {record.unit_id: record for record in records}
    return int(dict(by_unit[alarm.unit_id].tags)["step"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    specs = [
        RunSpec(
            name=entry["name"],
            arm=Arm(entry["arm"]),
            directory=Path(entry["directory"]),
            parent=None if entry.get("parent") is None else Path(entry["parent"]),
            fork_step=entry.get("fork_step"),
            onset=entry.get("onset"),
        )
        for entry in json.loads(arguments.manifest.read_text())
    ]
    results = [evaluate_run(spec) for spec in specs]
    arguments.output.write_text(
        "\n".join(["# Mid-run benchmark results", "", *render(results)]) + "\n"
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
