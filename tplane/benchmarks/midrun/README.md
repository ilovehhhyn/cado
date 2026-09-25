# Mid-run integrity benchmark

This benchmark tests `tp integrity` on real RL training runs whose faults start partway through. It checks three things:
- whether each configuration failure is detected quickly;
- whether a reward drop is attributed correctly, to configuration or to the algorithm;
- that healthy training raises no alarm.

The plan, the desired behaviours D1 to D8 and the arms are in `docs/superpowers/plans/2026-09-25-midrun-benchmark.md`.

The substrate is `ariahw/rl-rewardhacking-ext` at commit 0938cb6: verl 0.6.1, Qwen3-4B with LoRA, GRPO, and the `leetcode_rh` coding environment.

## Pieces

| File | Role |
|---|---|
| `hooks.py` | Fault schedule, fault injection and record building. Pure and tested. |
| `verl_bridge.py` | The thin glue that the patched trainer calls on the driver process. |
| `apply_patch.py` | Makes anchored, idempotent edits to the substrate and its bundled verl: the fault hooks, processed vLLM log-probs, and a learning-rate scaling call. |
| `analyze.py` | Stitches each fork onto its parent's history, runs the monitor, and scores D1 to D8. |
| `della/setup.sh` | One-time environment build on the Della login node. |
| `della/validate.sbatch` | Tiny run that reaches every phase once, before any long job. |

## How truth is kept from the monitor

The hooks write the per-sample ground truth (`healthy`, `sandbox_typed`, `sandbox_untyped`, `grader_missing`) to `truth-*.jsonl`, never into a record. The monitor reads only `records-*.jsonl`.

A masked failure gets the mean clean reward of its prompt group, so its GRPO advantage is zero.
