"""Check a validation job's outputs: every phase wrote records, the faults appear from their onset, and mismatch is finite."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from tplane.wire import parse_record


def records(directory: Path, step: int) -> list:
    path = directory / f"records-{step:05d}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; the run did not reach step {step}")
    return [parse_record(json.loads(line)) for line in path.read_text().splitlines()]


def main(root: Path) -> int:
    base = [records(root / "base", step) for step in (1, 2, 3, 4)]
    zeroed_before = sum(1 for record in base[1] if record.failure is not None)
    zeroed_after = sum(
        1 for record in base[2] + base[3] if record.failure is not None and record.reward == 0.0
    )
    if zeroed_before != 0 or zeroed_after == 0:
        raise SystemExit(
            f"sandbox_zeroed must start at step 3: {zeroed_before} failures before, {zeroed_after} after"
        )
    for name in ("base", "top_p", "lr"):
        steps = (1, 2, 3, 4) if name == "base" else (3, 4)
        for step in steps:
            train = [record for record in records(root / name, step) if record.mismatch is not None]
            if len(train) != 1 or not math.isfinite(train[0].mismatch.kl):
                raise SystemExit(
                    f"{name} step {step} must have one finite mismatch summary, got {train}"
                )
            print(
                f"{name} step {step}: kl {train[0].mismatch.kl:.3e} ratio {train[0].mismatch.ratio_mean:.4f} +- {train[0].mismatch.ratio_standard_error:.4f}"
            )
    if (root / "top_p" / "records-00002.jsonl").exists():
        raise SystemExit(
            "the top_p fork must start after the step-2 checkpoint, but it wrote step 2"
        )
    print(
        "VALIDATION_OK: compare the printed ratio of top_p steps 3-4 (expected below 1) with base; check actor/lr in the lr fork log"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
