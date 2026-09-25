"""Build one JSON cache per run: registry fields and per-step telemetry, without reading any fault-injection field.

Per step: the 8 train-step metrics, plus trajectory aggregates (mean and sd of reward, response length, advantage
sd, repetition ratio, unique-token ratio, tool-call and tool-error counts, and empty/short/long flag rates).
The fault_injection metadata is read only into the label block, never into features.
"""
import collections, csv, json, math, pathlib, statistics, sys
root = pathlib.Path("/scratch/gpfs/ARORA/hh9077/cado/bench/RFT-FaultBench-full")
out = pathlib.Path("/scratch/gpfs/ARORA/hh9077/cado/work/cache.jsonl")
registry = {}
for path in sorted((root / "metadata" / "registries").glob("*.csv")):
    for row in csv.DictReader(path.open()):
        registry[row["experiment_id"]] = row
TRAJ = ("reward", "response_length", "advantage_std", "repetition_ratio", "unique_token_ratio", "tool_call_count", "tool_error_count", "advantage_mean", "return")
FLAGS = ("empty_response", "short_response", "long_response")
def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values: return (None, None)
    return (statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0)
with out.open("w") as sink:
    for step_file in sorted(root.rglob("openrlhf_train_train_step.jsonl")):
        run = step_file.parent.parent
        reg = registry.get(run.name, {})
        steps = [json.loads(l) for l in step_file.read_text().splitlines() if l.strip()]
        per_step = collections.defaultdict(lambda: collections.defaultdict(list))
        injected = False
        with (step_file.parent / "openrlhf_train_trajectory.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                step = row["metadata"].get("rollout_step")
                fi = row["metadata"].get("fault_injection") or {}
                if fi.get("fault_injection_type") and fi.get("fault_injection_strength") not in (None, 0, 0.0, "0"):
                    injected = True
                for key in TRAJ:
                    per_step[step][key].append(row["metrics"].get(key))
                for key in FLAGS:
                    per_step[step][key].append(1.0 if row["flags"].get(key) else 0.0)
        trajectory = {}
        for step, columns in sorted(per_step.items(), key=lambda item: (item[0] is None, item[0] or 0)):
            entry = {"count": len(columns["reward"])}
            for key in TRAJ:
                mean, sd = stats(columns[key]); entry[key + "_mean"] = mean; entry[key + "_sd"] = sd
            for key in FLAGS:
                entry[key + "_rate"] = stats(columns[key])[0]
            trajectory[str(step)] = entry
        sink.write(json.dumps({
            "run": run.name, "group": run.parent.name,
            "label": {"registry": reg.get("control_or_fault", "baseline"), "injected": injected,
                      "fault_type": reg.get("fault_type"), "family": reg.get("fault_family"),
                      "task_group": reg.get("task_group") or ("arith_b" if "arith_b" in run.name else "arith_a")},
            "steps": [s["metrics"] for s in steps],
            "trajectory": trajectory,
        }) + "\n")
        print(run.name, len(steps), file=sys.stderr)
